import sqlite3
## Connect to db file
# connection = sqlite3.connect("daily_digest.db")
# cursor = connection.cursor()
# ## Create a table
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS user_channels (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_id INTEGER NOT NULL,
#         channel_id INTEGER NOT NULL,
#         channel_name TEXT NOT NULL,
#         status BOOLEAN NOT NULL DEFAULT TRUE
#     )
# """)
# connection.commit()

def insert_user_channel(user_id, channel_id, channel_name):
     with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()
        query="INSERT INTO user_channels (user_id, channel_id, channel_name) VALUES (?, ?, ?)"
        values = (user_id, channel_id, channel_name)
        cursor.execute(query, values)
        conn.commit()
        print(f"Inserted: User ID={user_id}, Channel ID={channel_id}")

def change_status(user_id, channel_id): 
     with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()
        query = "UPDATE user_channels SET status=NOT status WHERE user_id=? AND channel_id=?"
        values = (user_id, channel_id)
        cursor.execute(query, values)
        conn.commit()
        print(f"Changed status for user {user_id} and channel {channel_id}")

def unsubscribe_by_id(user_id, channel_id): 
     with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()
        query = "UPDATE user_channels SET status=NOT status WHERE user_id=? AND channel_id=?"
        values = (user_id, channel_id)
        cursor.execute(query, values)
        conn.commit()
        print(f"Changed status for user {user_id} and channel {channel_id}")

# def my_channel_names(user_id):
#      with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
#         cursor = conn.cursor()
#         query = "SELECT channel_id, channel_name FROM user_channels WHERE user_id=? AND status=1"
#         values = (user_id, )
#         cursor.execute(query, values)
#         rows = cursor.fetchall()
#         return [row[1] for row in rows]
    
# def my_channel_ids(user_id):
#      with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
#         cursor = conn.cursor()
#         query = "SELECT channel_id, channel_name FROM user_channels WHERE user_id=? AND status=1"
#         values = (user_id, )
#         cursor.execute(query, values)
#         rows = cursor.fetchall()
#         return [row[0] for row in rows]

def get_user_channels(user_id, type):
    with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()
        
        if type == 'all':
            query = "SELECT channel_name, channel_id, status  FROM user_channels WHERE user_id=?"
            values = (user_id, )

        elif type == 'active': 
            query = "SELECT channel_name, channel_id, status  FROM user_channels WHERE user_id=? AND status=1"
            values = (user_id, )

        elif type == 'inactive': 
            query = "SELECT channel_name, channel_id, status  FROM user_channels WHERE user_id=? AND status=0"
            values = (user_id, )
        else:
            raise ValueError("Invalid type. Use 'all', 'active', or 'inactive'.")

        # Execute and fetch results
        cursor.execute(query, values)
        rows = cursor.fetchall()
        return [list(row) for row in rows]

def get_channel_status(user_id, channel_id): 
    with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()

        query = "SELECT status FROM user_channels WHERE user_id=? AND channel_id=?"
        values = (user_id, channel_id)
        cursor.execute(query, values)
        result =cursor.fetchone()
        if result is None:
            return None
        return result[0]

def get_channel_name_by_id(channel_id): 
    with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()

        query = "SELECT channel_name FROM user_channels WHERE channel_id=?"
        values = (channel_id, )
        cursor.execute(query, values)
        result =cursor.fetchone()
        if result is None:
            return None
        return result[0]

def get_all_users(): 
    with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()
        query = "SELECT DISTINCT user_id FROM user_channels WHERE status = 1"
        cursor.execute(query)
        results =cursor.fetchall()
        if results is None:
            return None
        return [list(result)[0] for result in results]


def get_all_data():
    with sqlite3.connect("./telegram_bot/helpers/daily_digest.db") as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM user_channels"
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
