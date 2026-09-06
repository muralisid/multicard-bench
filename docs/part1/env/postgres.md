# PostgreSQL 17 with pgvector, local user-space install

Set up on 2026-09-06 on the Apple Silicon Mac. Everything runs as the current user. No sudo, no brew services, no launchd.

## Versions

- PostgreSQL 17.11 (Homebrew bottle, arm64, keg-only formula postgresql@17)
- pgvector 0.8.6 (Homebrew bottle)
- Homebrew 6.0.19

The pgvector formula declares build deps on postgresql@17 and postgresql@18. The bottle was poured, so postgresql@18 was not installed. Only postgresql@17 is present.

## Paths

- Bin dir (keg-only, not on PATH): /opt/homebrew/opt/postgresql@17/bin
- Data dir (cluster): /Users/muralisid/github_other/part1-tools/pgdata
- Server log: /Users/muralisid/github_other/part1-tools/env/postgres.log
- Pid file (copy of the first line of postmaster.pid): /Users/muralisid/github_other/part1-tools/env/postgres.pid
- pgvector shared library, as seen by pg17: /opt/homebrew/lib/postgresql@17/vector.dylib (symlink into /opt/homebrew/Cellar/pgvector/0.8.6/)
- pgvector extension SQL files: /opt/homebrew/share/postgresql@17/extension/vector*
- Unix socket: /tmp/.s.PGSQL.5433

Homebrew also created a default cluster at /opt/homebrew/var/postgresql@17 during install. It is not used and not started. Ignore it.

## Cluster settings

- initdb options: --locale=C -E UTF8 -U muralisid
- Port: 5433 (set in postgresql.conf, appended at the bottom)
- listen_addresses: localhost
- Auth (pg_hba.conf, initdb defaults): trust for local socket and for 127.0.0.1 and ::1. The pgr password is set on the role but is not checked for local connections. Local dev only.
- Superuser: muralisid (the OS user). Connect with psql -h 127.0.0.1 -p 5433 -d postgres for admin work.

## Role and database

- Role: pgr, LOGIN, password pgr
- Database: pgr, owner pgr
- Extension vector 0.8.6 created in database pgr

## Connection URL

postgresql://pgr:pgr@127.0.0.1:5433/pgr

## Start command

LC_ALL must be set. Without it the server dies at startup with "postmaster became multithreaded during startup" (a known macOS issue). LC_ALL=C matches the cluster locale.

    LC_ALL=C /opt/homebrew/opt/postgresql@17/bin/pg_ctl -D /Users/muralisid/github_other/part1-tools/pgdata -l /Users/muralisid/github_other/part1-tools/env/postgres.log -w start
    head -1 /Users/muralisid/github_other/part1-tools/pgdata/postmaster.pid > /Users/muralisid/github_other/part1-tools/env/postgres.pid

pg_ctl start detaches on its own. No nohup needed.

## Stop command

    /opt/homebrew/opt/postgresql@17/bin/pg_ctl -D /Users/muralisid/github_other/part1-tools/pgdata -m fast -w stop

## Health check

    /opt/homebrew/opt/postgresql@17/bin/pg_isready -h 127.0.0.1 -p 5433

Expected: "127.0.0.1:5433 - accepting connections"

Each pg_isready call writes one line to postgres.log: FATAL: database "muralisid" does not exist. That is how pg_isready works. It is harmless.

## State at the end of setup

Server RUNNING, pid 50337 (after one verified stop/start round trip). Data dir uses 46 MB. Table vec_smoke with 4 rows is left in database pgr.

## Verification output

Run through the connection URL above.

    SELECT version();
    PostgreSQL 17.11 (Homebrew) on aarch64-apple-darwin24.6.0, compiled by Apple clang version 17.0.0 (clang-1700.6.4.2), 64-bit

    SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
     extname | extversion
    ---------+------------
     vector  | 0.8.6

    CREATE TABLE vec_smoke (id serial PRIMARY KEY, label text, emb vector(3));
    INSERT INTO vec_smoke (label, emb) VALUES ('a', '[1,0,0]'), ('b', '[0,1,0]'), ('c', '[1,1,0]'), ('d', '[0.9,0.1,0]');
    SELECT label, emb, round((emb <=> '[1,0,0]')::numeric, 4) AS cosine_distance
      FROM vec_smoke ORDER BY emb <=> '[1,0,0]' LIMIT 4;

     label |     emb     | cosine_distance
    -------+-------------+-----------------
     a     | [1,0,0]     |          0.0000
     d     | [0.9,0.1,0] |          0.0061
     c     | [1,1,0]     |          0.2929
     b     | [0,1,0]     |          1.0000

The <=> operator is pgvector cosine distance. Order is correct: identical vector first, orthogonal vector last.

After a stop and start, SELECT count(*) FROM vec_smoke returned 4. Data persists.

## Python client note

No Python driver was installed by this stage. psycopg (v3) or asyncpg would need adding to the bench venv if the next stage wants Python access. Not done here.
