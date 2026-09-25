import os
import hashlib
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

# FIX: Fetch keys from .env, OR fallback directly to local MongoDB if missing
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "urdu_speech_db")

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        try:
            # Local MongoDB does not require server_api version 1
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            _db = _client[DB_NAME]
            _client.admin.command('ping')
            print("[DB] Local MongoDB connected successfully!")
        except Exception as e:
            print(f"[DB] MongoDB connection error: {e}")
    return _db


def close_db():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


# =====================
# User Operations
# =====================

def create_user(name, email, password_hash, role="user"):
    db = get_db()
    existing = db.users.find_one({"email": email})
    if existing:
        return None
    user = {
        "name": name,
        "email": email,
        "password": password_hash,
        "role": role,
        "created_at": datetime.utcnow(),
        "analyses_count": 0,
        "avg_stress": "N/A",
    }
    result = db.users.insert_one(user)
    user["_id"] = result.inserted_id
    return user


def find_user_by_email(email):
    db = get_db()
    return db.users.find_one({"email": email})


def find_user_by_id(user_id):
    db = get_db()
    from bson import ObjectId
    return db.users.find_one({"_id": ObjectId(user_id)})


def update_user_stats(user_id, stress_level):
    db = get_db()
    from bson import ObjectId
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$inc": {"analyses_count": 1}}
    )


# =====================
# Analysis Report Operations
# =====================

def save_analysis_report(user_id, report_data):
    db = get_db()
    report = {
        "user_id": user_id,
        "emotion": report_data.get("emotion"),
        "stress_level": report_data.get("stress_level"),
        "lie_probability": report_data.get("lie_probability"),
        "lie_percentage": report_data.get("lie_percentage"),
        "transcription": report_data.get("transcription"),
        "ml_confidence": report_data.get("ml_confidence"),
        "features": report_data.get("features"),
        "audio_duration": report_data.get("audio_duration"),
        "audio_size_kb": report_data.get("audio_size_kb"),
        "created_at": datetime.utcnow(),
    }
    result = db.reports.insert_one(report)
    report["_id"] = result.inserted_id
    return report


def get_user_reports(user_id, limit=10):
    db = get_db()
    reports = db.reports.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
    return list(reports)


def get_all_reports(limit=50):
    db = get_db()
    reports = db.reports.find().sort("created_at", -1).limit(limit)
    return list(reports)


# =====================
# Admin Stats
# =====================

def get_admin_stats():
    db = get_db()
    total_users = db.users.count_documents({})
    total_analyses = db.reports.count_documents({})
    
    pipeline = [{"$group": {"_id": "$emotion", "count": {"$sum": 1}}}]
    emotion_dist = list(db.reports.aggregate(pipeline))
    
    stress_pipeline = [{"$group": {"_id": "$stress_level", "count": {"$sum": 1}}}]
    stress_dist = list(db.reports.aggregate(stress_pipeline))
    
    return {
        "total_users": total_users,
        "total_analyses": total_analyses,
        "emotion_distribution": {item["_id"]: item["count"] for item in emotion_dist if item["_id"]},
        "stress_distribution": {item["_id"]: item["count"] for item in stress_dist if item["_id"]},
    }


def get_all_users():
    db = get_db()
    users = db.users.find({}, {"password": 0}).sort("created_at", -1)
    return list(users)


def update_user_password(email, new_password_hash):
    db = get_db()
    result = db.users.update_one(
        {"email": email},
        {"$set": {"password": new_password_hash}}
    )
    return result.modified_count > 0


def save_contact_message(name, email, subject, message):
    db = get_db()
    msg = {
        "name": name,
        "email": email,
        "subject": subject,
        "message": message,
        "created_at": datetime.utcnow(),
        "read": False,
    }
    result = db.contact_messages.insert_one(msg)
    msg["_id"] = result.inserted_id
    return msg


def get_contact_messages(limit=50):
    db = get_db()
    messages = db.contact_messages.find().sort("created_at", -1).limit(limit)
    return list(messages)


def ensure_admin_exists():
    db = get_db()
    default_email = "admin@urduspeech.ai"
    default_pass_hash = hashlib.sha256("admin123".encode()).hexdigest()

    default_admin = db.users.find_one({"email": default_email})
    if default_admin:
        db.users.update_one(
            {"email": default_email},
            {"$set": {"password": default_pass_hash, "role": "admin"}}
        )
        print("[DB] Admin account verified: admin@urduspeech.ai / admin123")
    else:
        admin_user = {
            "name": "Admin",
            "email": default_email,
            "password": default_pass_hash,
            "role": "admin",
            "created_at": datetime.utcnow(),
            "analyses_count": 0,
            "avg_stress": "N/A",
        }
        db.users.insert_one(admin_user)
        print("[DB] Default admin created: admin@urduspeech.ai / admin123")