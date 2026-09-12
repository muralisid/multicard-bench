"""Separate topic/community indexes and a typed graph of discovered evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
import random
import re
import time

import igraph as ig
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import CONFIG
from .facts import norm


class EvidenceGraph:
    def __init__(self, documents, facts):
        started = time.perf_counter()
        self.docs = {d["id"]: d for d in documents}
        self.ids = sorted(self.docs)
        self.position = {uid: i for i, uid in enumerate(self.ids)}
        self.topics, self.communities, self.entities = defaultdict(list), defaultdict(list), defaultdict(set)
        self.doc_topics, self.doc_communities, self.adj = {}, {}, defaultdict(set)
        self.nodes, self.edges = [], []
        self.vectorizer, self.matrix = None, None
        if self.ids:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
            try:
                self.matrix = self.vectorizer.fit_transform([self.docs[uid]["text"] for uid in self.ids])
            except ValueError:
                self.vectorizer = None
        if self.matrix is not None:
            k = min(len(self.ids), max(1, min(32, int(math.sqrt(len(self.ids) / 4)))))
            model = MiniBatchKMeans(n_clusters=k, random_state=13, n_init=3, batch_size=256)
            labels = model.fit_predict(self.matrix)
            words = self.vectorizer.get_feature_names_out()
            for index in range(k):
                label = ", ".join(words[i] for i in np.argsort(-model.cluster_centers_[index])[:5])
                self.nodes.append({"id": f"topic:{index}", "type": "topic", "label": label})
            for uid, label in zip(self.ids, labels):
                label = int(label)
                self.doc_topics[uid] = label
                self.topics[label].append(uid)
                self.edges.append(("doc:" + uid, "belongs_to_topic", f"topic:{label}"))
        for fact in facts:
            source = fact["unit_id"]
            if source not in self.docs:
                continue
            subject, obj = norm(fact["subject"]), norm(fact["object"])
            self.entities[subject].add(source)
            self.entities[obj].add(source)
            self.adj[subject].add(obj)
            self.adj[obj].add(subject)
            relation = "fact:" + fact["id"]
            self.nodes.append({"id": relation, "type": "fact", "predicate": fact["predicate"],
                               "negated": fact.get("negated", False), "valid_from": fact.get("valid_from"),
                               "valid_to": fact.get("valid_to"), "superseded_by": fact.get("superseded_by")})
            self.edges.extend([(relation, "subject", "entity:" + subject),
                               (relation, "object", "entity:" + obj),
                               (relation, "supported_by", "doc:" + source)])
        weights = Counter()
        for entity, docs in sorted(self.entities.items()):
            self.nodes.append({"id": "entity:" + entity, "type": "entity", "label": entity})
            self.edges.extend(("doc:" + uid, "mentions", "entity:" + entity) for uid in sorted(docs))
            # Prevent common "user" nodes creating a complete document graph.
            if len(docs) <= min(64, max(2, len(self.ids) // 2)):
                for pair in combinations(sorted(self.position[uid] for uid in docs), 2):
                    weights[pair] += 1
        graph = ig.Graph(n=len(self.ids), edges=list(weights))
        ig.set_random_number_generator(random.Random(13))
        membership = (graph.community_leiden(objective_function="modularity", weights=list(weights.values()),
                                            n_iterations=2).membership if weights else list(range(len(self.ids))))
        for uid, label in zip(self.ids, membership):
            self.doc_communities[uid] = label
            self.communities[label].append(uid)
            self.nodes.append({"id": "doc:" + uid, "type": "source"})
            self.edges.append(("doc:" + uid, "belongs_to_community", f"community:{label}"))
        self.nodes.extend({"id": f"community:{i}", "type": "community"} for i in sorted(self.communities))
        self.seconds = time.perf_counter() - started

    def rank(self, question, base):
        if self.matrix is None:
            return base, {"topics": [], "communities": [], "relations": []}
        sims = np.asarray((self.matrix @ self.vectorizer.transform([question]).T).toarray()).ravel()
        seeds = [uid for uid in base[:5] if uid in self.position]
        topic_pool = {uid for seed in seeds for uid in self.topics[self.doc_topics[seed]]}
        community_pool = {uid for seed in seeds for uid in self.communities[self.doc_communities[seed]]}
        query = norm(question)
        entities = {e for e in self.entities if len(e) >= 3 and re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", query)}
        if re.search(r"\b(i|me|my|mine)\b", query) and "user" in self.entities:
            entities.add("user")
        for _ in range(2):
            entities |= {neighbor for e in list(entities) for neighbor in self.adj[e]}
        relation_pool = {uid for e in entities for uid in self.entities[e]}
        def ordered(pool):
            return sorted((uid for uid in pool if sims[self.position[uid]] > 0),
                          key=lambda uid: (-sims[self.position[uid]], uid))[:100]
        channels = {"base": base, "topics": ordered(topic_pool), "communities": ordered(community_pool),
                    "relations": ordered(relation_pool)}
        scores = Counter()
        for channel, ranking in channels.items():
            for rank, uid in enumerate(ranking, 1):
                scores[uid] += CONFIG["weights"][channel] / (CONFIG["rrf_k"] + rank)
        return sorted(scores, key=lambda uid: (-scores[uid], uid)), {k: v for k, v in channels.items() if k != "base"}

    def artifact(self):
        return {"nodes": self.nodes, "edges": self.edges, "topics": dict(self.topics),
                "communities": dict(self.communities), "build_seconds": self.seconds,
                "scope": "discovered evidence within one corpus group"}
