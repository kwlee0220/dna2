import psycopg2 as pg2

conn = pg2.connect("host=localhost dbname=dna user=postgres password=dna2021 port=5432")
conn.autocommit = True

cur = conn.cursor()
cur.execute("select * from track_events")
rows = cur.fetchall()
conn.commit()

print(rows)

"""
create table track_events (
    camera_id varchar not null,
    luid varchar not null,
    bbox box not null,
    frame_index bigint not null,
    ts timestamp not null,

    constraint track_events_pkey primary key (camera_id, luid)
);
"""