from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# Correct DB URI (no extra quotes, no wrong formatting)
DB_URI = "postgresql://neondb_owner:npg_rAgnONmIP50C@ep-blue-wind-adxyifwj-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Create connection pool
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0}
)

# Setup PostgresSaver
checkpointer = PostgresSaver(pool) # 
checkpointer.setup()  # this will create the required tables