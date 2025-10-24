import streamlit as st
import os
import json
import time
import base64
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# Local modules (assuming they exist)
from file_loader import get_raw_text
from qa_engine import build_qa_engine, save_vectorstore, load_vectorstore

# ---------------- Load Environment ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# ---------------- Environment Variables ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") 
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# ---------------- Session State Defaults ----------------
defaults = {
    "page_initialized": False,
    "chat_history": [],
    "qa": None,
    "current_cache_name": None,
    "page": "chat",  # Start with chat
    "authorized_settings": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Global CSS ----------------
GLOBAL_CSS = """
<style>
# .header-bar {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         background-color: #0073e6;
#         padding: 5px 15px;
#         border-radius: 8px;
#         margin-bottom: 20px;
# }
.header-box {
    background-color: white;
    color: #0073e6;
    padding: 5px 15px;
    border-radius: 8px;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    user-select: none;
    }
.header-title {
    font-size: 30px;
    font-weight: bold;
}
</style>
"""
st.set_page_config(page_title="Intelligent Assistant", layout="wide")

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------- Utility Functions ----------------
def render_header():
    """ Renders the top header with cache selection and settings access """
    # st.markdown('<div class="header-bar">', unsafe_allow_html=True)
    st.markdown('<div class="header-box">Intelligent Assistant</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 6, 2])

    with c1:
        caches = ["-- select --"] + list_caches()
        selected = st.selectbox(
            "Hub",
            options=caches,
            index=caches.index(st.session_state.current_cache_name) if st.session_state.current_cache_name in caches else 0,
            key="header_cache_select"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:

        if selected and selected != "-- select --":
            if selected != st.session_state.current_cache_name:
                if load_cache_into_memory(selected):
                    st.session_state.chat_history = []
                    st.rerun()

 
        settings_clicked = st.button("⚙️ Settings", key="go_settings", help="Open settings", use_container_width=True)
        if settings_clicked:
            st.session_state.page = "settings"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Persistence ----------------
BASE_DIR = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(BASE_DIR, "persisted_data")

# Create the directory if it doesn't exist
os.makedirs(PERSIST_DIR, exist_ok=True)

def list_caches() -> List[str]:
    try:
        # Check if the directory exists
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR, exist_ok=True)
        # List all directories inside PERSIST_DIR
        return [d for d in os.listdir(PERSIST_DIR) if os.path.isdir(os.path.join(PERSIST_DIR, d))]
    except Exception as e:
        # Log the error
        st.error(f"Error while listing caches: {e}")
        return []
    
# ---------------- SharePoint Helpers ----------------
def get_graph_token() -> str:
    if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
        raise RuntimeError("SharePoint credentials not set in .env.")
    token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default"
    }
    token_response = requests.post(token_url, data=token_data)
    token_response.raise_for_status()
    return token_response.json()["access_token"]


def share_link_to_drive_item_meta(share_link: str, access_token: str) -> Dict[str, Any]:
    encoded_url = base64.urlsafe_b64encode(share_link.strip().encode("utf-8")).decode("utf-8").rstrip("=")
    meta_url = f"https://graph.microsoft.com/v1.0/shares/u!{encoded_url}/driveItem"
    meta_res = requests.get(meta_url, headers={"Authorization": f"Bearer {access_token}"})
    meta_res.raise_for_status()
    return meta_res.json()

def list_children_for_item(item_id: str, access_token: str) -> List[Dict[str, Any]]:
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id_from_item(item_id, access_token)}/items/{item_id}/children"
    res = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
    res.raise_for_status()
    return res.json().get("value", [])


def drive_id_from_item(item_id: str, access_token: str) -> str:
    # Retrieve driveId for given itemId
    meta_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}"
    url = f"https://graph.microsoft.com/v1.0/drive/items/{item_id}"
    res = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
    res.raise_for_status()
    parent = res.json().get("parentReference", {})
    drive_id = parent.get("driveId")
    if not drive_id:
        # fallback: root drive
        drive_id = res.json().get("parentReference", {}).get("driveId", "")
    return drive_id


def collect_files_recursively_from_item(item_json: Dict[str, Any], access_token: str) -> List[Dict[str, Any]]:
    results = []

    def _walk(item):
        if "file" in item:
            results.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "etag": item.get("eTag") or item.get("@microsoft.graph.downloadUrl", "")[:32],
                "size": item.get("size"),
                "lastModifiedDateTime": item.get("lastModifiedDateTime"),
                "downloadUrl": item.get("@microsoft.graph.downloadUrl"),
            })
        elif "folder" in item:
            # fetch children of this folder
            drive_id = item.get("parentReference", {}).get("driveId")
            current_id = item.get("id")
            if not drive_id:
                drive_id = drive_id_from_item(current_id, access_token)
            children_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{current_id}/children"
            res = requests.get(children_url, headers={"Authorization": f"Bearer {access_token}"})
            res.raise_for_status()
            for child in res.json().get("value", []):
                _walk(child)

    _walk(item_json)
    return results


def build_manifest(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Map id -> etag for delta detection; also keep readable list
    return {
        "files": [
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "etag": f.get("etag"),
                "size": f.get("size"),
                "lastModifiedDateTime": f.get("lastModifiedDateTime"),
            } for f in files
        ],
        "map": {f.get("id"): f.get("etag") for f in files},
        "count": len(files)
    }


def manifests_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a.get("map") == b.get("map")) and (a.get("count") == b.get("count"))

def download_and_extract_text(files: List[Dict[str, Any]]) -> str:
    all_text = ""
    for f in files:
        url = f.get("downloadUrl")
        if not url:
            continue
        fres = requests.get(url)
        fres.raise_for_status()
        all_text += get_raw_text(fres.content, f.get("name", "file")) + "\n\n"
    return all_text

def load_manifest(cache_name: str) -> list:
    try:
        manifest_path = os.path.join(PERSIST_DIR, cache_name, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                data = json.load(f)
                return data.get("files", [])
        return []
    except Exception as e:
        st.error(f"Error loading manifest: {e}")
        return []
    
def chat_input():
    """ Input field for chatting """
    return st.text_input("Ask something...", key="chat_input", placeholder="Type your question here...", label_visibility="collapsed")

# ---------------- Chat Tab ----------------
def page_chat():
    render_header()
    
    # Initialize files variable
    files = []
    
    # Show current cache info and files (collapsed by default)
    if st.session_state.current_cache_name:
        files = load_manifest(st.session_state.current_cache_name)
    with st.expander("📁 Files in Memory", expanded=False):
        if files:
            for f in files:
                st.markdown(f"- 🗂️ **{f['name']}**")
        else:
            st.info("No files found in this memory.")

                
    # Show chat history
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            # User → right
            with st.chat_message("user" , avatar="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg"):
                # st.markdown(chat["question"])
             st.markdown(f"<div style='text-align:right'>{chat['question']}</div>", unsafe_allow_html=True)
            # Assistant → left
            with st.chat_message("assistant", avatar="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg"):
                # st.markdown(chat["answer"])
             st.markdown(f"<div style='text-align:left'>{chat['answer']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
                
    # Chatbox container
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:20px; font-weight:bold;">Chat with your Document</h3>', unsafe_allow_html=True)
    user_query = chat_input()

    # Handle user input and chat history
    if user_query:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa({"query": user_query})
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": result.get("result", ""),
                    "context": result.get("source_documents", [])
                })
            except Exception as e:
                st.error(f"Query error: {e}")

                
# ---------------- QA Loader/Builder ----------------
def rebuild_vectorstore_and_save(cache_name: str, raw_text: str):
    try:
        # Rebuild the QA engine and vectorstore
        qa, vectorstore = build_qa_engine(
            raw_text=raw_text,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            cache_name=cache_name
        )
        
        # Save the vectorstore to disk
        save_vectorstore(vectorstore, PERSIST_DIR, cache_name=cache_name)
        
        # Update session state
        st.session_state.qa = qa
        st.session_state.current_cache_name = cache_name
        
        # Provide success message
        st.success(f"Cache '{cache_name}' rebuilt and saved successfully.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

def load_cache_into_memory(name: str):
    try:
        vectorstore = load_vectorstore(OPENAI_API_KEY, PERSIST_DIR, cache_name=name, openai_api_base=OPENAI_API_BASE)
        if vectorstore:
            st.session_state.qa, _ = build_qa_engine(
                raw_text="",
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                load_vectorstore_obj=vectorstore
            )
            st.session_state.current_cache_name = name
            
            # ✅ Auto-update manifest if missing or outdated
            manifest_path = os.path.join(PERSIST_DIR, name, "manifest.json")
            if not os.path.exists(manifest_path):
                st.info("Manifest not found. Attempting to rebuild...")
                # Try to fetch SharePoint link metadata if known (optional)
                # Or just create a minimal manifest so the Files in Memory section doesn’t break
                manifest = {"files": [], "map": {}, "count": 0}
                os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                    
            return True
    except Exception as e:
        st.error(f"Failed to load cache '{name}': {e}")
    return False
# ---------------- Settings Tab ----------------
def page_settings():
    render_header()
    st.header("Settings")

    if not st.session_state.authorized_settings:
        st.warning("Authorization required to access settings.")
        
        if st.button("⬅️ Back to Chat"):
           st.session_state.page = "chat"
           st.rerun()
        
        pwd = st.text_input("Admin password", type="password")
        if st.button("Unlock"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.authorized_settings = True
                st.success("Authorized")
                st.rerun()
            else:
                st.error("Invalid password")
        return

    tabs = st.tabs(["Upload File", "Load from SharePoint", "Memory Management"])

    with tabs[0]:
        st.subheader("Upload a Document")
        uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, XLS, XLSX, ZIP)", type=["pdf", "docx", "xls", "xlsx", "zip"])
        # cache_name = st.text_input("Cache Name (unique)")
        cache_name = st.text_input("Hub (unique)")

        if st.button("Process & Save to Memory"):
            if uploaded_file and cache_name:
                try:
                    with st.spinner("Processing..."):
                        raw_bytes = uploaded_file.read()
                        raw_text = get_raw_text(raw_bytes, uploaded_file.name)
                        if raw_text.strip():
                            rebuild_vectorstore_and_save(cache_name, raw_text)
                            st.success(f"Saved knowledge base as '{cache_name}'.")
                        else:
                            st.error("No text extracted from the file.")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    with tabs[1]:
        st.subheader("SharePoint Integration")
        sp_link = st.text_input("SharePoint File/Folder Link")
        cache_name_sp = st.text_input("Hub Name for SharePoint")
        autosync = st.checkbox("Enable Auto-Sync")

        if st.button("Load from SharePoint"):
            if sp_link and cache_name_sp:
                try:
                    with st.spinner("Loading..."):
                        token = get_graph_token()
                        item_json = share_link_to_drive_item_meta(sp_link, token)
                        files = collect_files_recursively_from_item(item_json, token)
                        text = download_and_extract_text(files)
                        rebuild_vectorstore_and_save(cache_name_sp, text)
                        manifest = build_manifest(files)
                        manifest_path = os.path.join(PERSIST_DIR, cache_name_sp, "manifest.json")
                        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)
                        st.success(f"Loaded SharePoint data as '{cache_name_sp}'.")
                
                # ✅ Auto-sync feature
                    if autosync:
                       st.info("Auto-sync enabled. Checking for updates every 30 seconds...")

                    while autosync:
                        time.sleep(30)
                        new_token = get_graph_token()
                        new_item_json = share_link_to_drive_item_meta(sp_link, new_token)
                        new_files = collect_files_recursively_from_item(new_item_json, new_token)
                        new_manifest = build_manifest(new_files)

                        if not manifests_equal(manifest, new_manifest):
                            st.info("🔄 Changes detected in SharePoint. Updating memory...")
                            new_text = download_and_extract_text(new_files)
                            rebuild_vectorstore_and_save(cache_name_sp, new_text)
                            with open(manifest_path, "w") as f:
                                json.dump(new_manifest, f, indent=2)
                            manifest = new_manifest
                            st.success("Memory auto-updated successfully!")
                        else:
                            st.write("✅ No changes detected.") 
                                  
                except Exception as e:
                        st.error(f"Error loading from SharePoint: {e}")

    with tabs[2]:
        st.subheader("Manage Memory")
        selected_cache = st.selectbox("Select Hub", options=list_caches())
        if selected_cache and st.button("Load Cache into Memory"):
            load_cache_into_memory(selected_cache)
            st.success(f"Loaded '{selected_cache}' into memory.")
            
        if st.button("Clear Memory"):
            st.session_state.qa = None
            st.session_state.chat_history = []
            st.session_state.current_cache_name = None
            st.success("Memory cleared.")
            
 # ------------- Back to Chat Button -------------
    if st.button("⬅️ Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()  # Rerun to switch the page back to chat

# ---------------- Router ----------------
if st.session_state.page == "chat":
    page_chat()
else:
    page_settings()


