# post-graph-rag 1.12.0: the Postgres schema it creates

Dumped on 2026-09-06 from the smoke database pgr_smoke (PostgreSQL 17.11, pgvector 0.8.6) after indexing three synthetic documents. Config: schema_per_realm=True, realm "smoke", embedding_dim 1536, embed_relations True (the default). The raw psql output is in pgr-schema-raw.txt next to this file. The database was dropped after the dump.

The tables are created by post-graph 1.5.0 (client_asyncpg.py, create_vertex_table and create_edge_table), called from post_graph_rag/graph_store.py initialize_schema (lines 49-125). post-graph-rag adds three things of its own: the unique entity name index (graph_store.py 159-176), the payload expression indexes on t_created, t_expired, dormant_since and level (graph_store.py 113-122), and the lexical GIN index on relations, built on first query (graph_store.py 567-608).

## Where things live

With schema_per_realm=True every realm is one Postgres schema named after the realm. Table names are plain (documents, entities, ...). With the default schema_per_realm=False all realms share one set of tables in the connecting user's search path and are told apart by the realm column. The evaluation harness uses schema_per_realm=True with one realm per LongMemEval instance and drops the schema afterwards (evaluation/longmemeval/run.py lines 347-349 and 418).

Almost every field post-graph-rag cares about is a key inside the jsonb payload column, not a real column. The only real columns are the ids, the space, the embedding, the timestamps, the edge endpoints and relation_type, and two generated columns derived from the payload.

## Tables (21 in the schema)

Seven graph tables, each with a shadow audit table and an append-only history table:

| Graph table | Kind | From | To | Audit table | History table |
|---|---|---|---|---|---|
| documents | vertex | | | documents_audit | documents_data |
| entities | vertex | | | entities_audit | entities_data |
| communities | vertex | | | communities_audit | communities_data |
| relations | edge | entities | entities | relations_audit | relations_data |
| doc_mentions | edge | documents | entities | doc_mentions_audit | doc_mentions_data |
| community_members | edge | communities | entities | community_members_audit | community_members_data |
| community_children | edge | communities | communities | community_children_audit | community_children_data |

Row counts after the three-document smoke: documents 3, entities 18, relations 48, doc_mentions 39, communities 0 (build_communities was not called), relations_audit 50, entities_audit 40, documents_audit 3. The history (_data) tables stayed empty. Database size 13 MB.

## Vertex table columns (documents, entities, communities)

| Column | Type | Notes |
|---|---|---|
| realm | text not null | tenant name, also the schema name here |
| id | bigint not null | sequence per table; primary key is (realm, id) |
| space | varchar(255) not null default 'default' | sub-tenant |
| uuid | uuid not null default gen_random_uuid() | unique index |
| fqid | text generated always as realm/table/id | stored |
| payload | jsonb not null default '{}' | all application fields, see below |
| created_at | timestamptz default CURRENT_TIMESTAMP | belief time for chunks (used by as_believed_at, engine.py 1029-1038) |
| updated_at | timestamptz default CURRENT_TIMESTAMP | kept fresh by a BEFORE UPDATE trigger |
| pt_valid_from | text generated always, stored | payload->>'valid_from' padded to YYYY-MM-DD |
| pt_valid_to | text generated always, stored | payload->>'valid_to' padded to YYYY-MM-DD |
| embedding | vector(1536) | pgvector; the width is fixed at table creation |

Indexes on each vertex table: primary key (realm, id); hnsw (embedding vector_cosine_ops); gin (payload); btree on pt_valid_from and pt_valid_to; btree (realm, space); unique (uuid). Plus on entities: the unique index smoke_entities_name_uniq on (realm, space, lower(payload->>'name')) and a btree on (payload->>'dormant_since'). Plus on communities: a numeric btree on (payload->>'level').

Triggers on each: audit_<table>_trigger AFTER INSERT OR DELETE OR UPDATE writes old_row and new_row jsonb into <table>_audit; update_<table>_modtime BEFORE UPDATE sets updated_at.

## Edge table columns (relations, doc_mentions, community_members, community_children)

Same as a vertex table plus:

| Column | Type | Notes |
|---|---|---|
| from_id | bigint not null | foreign key (realm, from_id) to the from table, ON DELETE CASCADE |
| to_id | bigint not null | foreign key (realm, to_id) to the to table, ON DELETE CASCADE |
| relation_type | text not null | the predicate (relations), "mentions" (doc_mentions), "includes" (community_members), "has_child" (community_children) |
| embedding | vector(1536) | relations only, and only because embed_relations is on; the other three edge tables have no embedding column |

Extra indexes on relations: hnsw on embedding; btree on (payload->>'t_created') and (payload->>'t_expired'); btree (realm, to_id); and after the first query the GIN index smoke_relations_fts on to_tsvector('english', relation_type || ' ' || payload->>'description'). No unique index on (from_id, relation_type, to_id); uniqueness of a triple is enforced in Python by find_relation before add_edge (graph_store.py 452-463, 491).

## Embedding columns and their width

All of these are vector(1536) in the smoke, one HNSW index each (cosine): documents.embedding, documents_data.embedding, entities.embedding, entities_data.embedding, communities.embedding, communities_data.embedding, relations.embedding, relations_data.embedding.

The width comes from RAGConfig.embedding_dim and is checked on every initialize() against the existing tables (graph_store.py 127-157). It cannot be 3072 on this Postgres: post-graph creates an HNSW index on every embedding column at table creation (post_graph/client_asyncpg.py lines 338 and 340), and pgvector 0.8.6 refuses "column cannot have more than 2000 dimensions for hnsw index" (verified in a rolled back transaction). Chandan's harness uses 1536. The smoke asked the proxy for 1536 with the OpenAI "dimensions" parameter, which litellm passes to Vertex as outputDimensionality; post-graph-rag itself never sends that parameter, so the smoke wraps LLMService (see pgr.md).

## Payload keys, as written by post-graph-rag

documents.payload (graph_store.py 180-196, engine.py 534-539):

| Key | Meaning |
|---|---|
| text | the chunk text |
| source, category, collection, document, page, paragraph, space | DocumentMetadata fields, only when set |
| doc_key | document identity: "source::document", else whichever is set, else "unkeyed" (models.py 92-111). Re-indexing the same key replaces the chunks (engine.py 338-352) |
| content_hash | first 32 hex chars of sha256 of the chunk text (models.py 114-116) |
| any extra keys | DocumentMetadata.extra is flattened in. The harness puts session_date here (run.py 162); the smoke put report_date |

Seen in the smoke: category, content_hash, doc_key, document, report_date, source, text (7 keys on every row).

entities.payload (graph_store.py 396-401, 434-439): name, type, description, aliases (json array). Optional: dormant_since (set when the last mentioning chunk is deleted, graph_store.py 289-329), revived_at, archived_at (retention, graph_store.py 1128-1153). Seen in the smoke: aliases, description, name, type.

relations.payload (graph_store.py 508-523 and 539-554):

| Key | Meaning |
|---|---|
| description | the relation's supporting sentence from the extractor |
| sources | json array of chunk ids (documents.id) that asserted this triple. This is the provenance of a relation. It is stored but not returned by query()/query_data() |
| weight | number of distinct source chunks |
| negated | true when the text denies the relation (the predicate stays positive) |
| confidence | 0 to 1 from the extractor, max over observations |
| valid_from, valid_to | validity in the world, as the text stated it, YYYY, YYYY-MM or YYYY-MM-DD, or null. Mirrored into the generated columns pt_valid_from and pt_valid_to, which the traversal filters on for as_of |
| t_created | transaction time: when this system first believed the relation (ISO UTC) |
| t_expired | when the system stopped believing it; set together with superseded_by |
| superseded_by | id of the newer relation that closed this one. Only written by supersede_conflicting (exclusive_predicate_groups) or mark_superseded (contradiction_detection) (graph_store.py 672-729 and 774-808). Absent otherwise; retrieval skips rows where it is set unless include_superseded |
| dormant_since, revived_at | set when every source chunk was deleted (graph_store.py 252-287) |

Seen in the smoke: confidence, description, negated, sources, t_created, t_expired, valid_from, valid_to, weight on all 48 rows. superseded_by never appeared because the smoke declared no exclusive_predicate_groups and contradiction_detection is off by default (same as the LongMemEval harness). 15 of 48 relations carried a validity date: 12 with valid_from, 6 with valid_to, 3 with both.

doc_mentions.payload is always {} (graph_store.py 825-834). The edge itself is the provenance link chunk -> entity.

communities.payload (graph_store.py 973-984): key, title, summary, level, rating, findings (array of {summary, explanation}), size, built_at. Not exercised in the smoke.

## Audit and history tables

<table>_audit: audit_id, realm, space, action (INSERT/UPDATE/DELETE), changed_by (from the session setting app.current_user_id, null here), changed_at, old_row jsonb, new_row jsonb. Written by the trigger on every row change, so every upsert of an entity or relation leaves a row (48 relations produced 50 audit rows: 48 inserts and 2 weight updates).

<table>_data: data_id, realm, id, space, payload, timestamp, embedding (vertex and relations history tables carry the embedding column). post-graph-rag does not write these in the smoke; they stayed at 0 rows.

## Functions

Two plpgsql trigger functions per schema: smoke.audit_trigger_func() and smoke.update_modified_column().

## How to read provenance for retrieval scoring

- A retrieved chunk carries chunk_id (documents.id) and metadata.doc_key, metadata.document, metadata.source. In the harness those are the session id (document=session_id, source=session://session_id).
- A retrieved relation carries edge_id (relations.id). Its source chunks are `SELECT payload->'sources' FROM <realm>.relations WHERE id = <edge_id>`; each entry is a documents.id, whose payload->>'document' is the session id.
- A retrieved entity carries only its name. Its chunks are `SELECT from_id FROM <realm>.doc_mentions WHERE to_id = <entity id>`; the entity id is not in the query output either, so join on lower(payload->>'name') within (realm, space), which is the unique key.
