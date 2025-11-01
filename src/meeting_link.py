import os
from dotenv import load_dotenv
from livekit import api

load_dotenv(".env.local")

human_token = (
    api.AccessToken(
        api_key=os.environ.get("LIVEKIT_API_KEY"),
        api_secret=os.environ.get("LIVEKIT_API_SECRET"),
    )
    .with_identity("human-user")
    .with_grants(api.VideoGrants(room_join=True, room="local-room"))
    .to_jwt()
)

meet_url = f"https://meet.livekit.io/custom?liveKitUrl=http://localhost:7880&token={human_token}"
print(meet_url)
