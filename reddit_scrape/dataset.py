import praw, prawcore
import csv
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import time
import argparse
import requests
from urllib.parse import urlparse
from praw.models import Comment
import vision_transcribe
import pandas as pd

# This affects how many of the top posts are being searched
# NOT how many values in final dataset
# NOTE: Found out that PRAW has a max limit of 1000
LIMIT = 1000
ID_LIST_PATH = "VerifiedIDS.csv"
DATASET_PATH = "AIODataVerified-small.csv"
IMAGE_DIR = "images"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIME_TO_EXT = {
    "image/jpg": ".jpg",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

DELAY = 180

# Unified dataset header used everywhere
DATASET_HEADER = (
    "Index",
    "SubmissionID",
    "URL",
    "Title",
    "Body",
    "Score",
    "UpvoteRatio",
    "BestComments",
    "TopComments",
    "ControversialComments",
    "QAComments",
    "Transcription",
)

# Get env vars
load_dotenv()
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
user_agent = os.getenv('user_agent')
username = os.getenv('reddit-username')
password = os.getenv('password')

################################
# General Helper Functions
################################

# Reddit API object setup
def get_reddit_obj():
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password
    )

    try:
        print("Logged in as:", reddit.user.me())
    except prawcore.OAuthException as e:
        print("OAuth failed:", e)
        exit(0)

    return reddit

# Overwrite or Appened to Output CSV
def write_tuples_to_csv(data, filename=DATASET_PATH, overwrite=False):
    if not data:
        raise ValueError("Data list is empty.")

    # Decide file mode based on overwrite flag
    mode = 'w' if overwrite else 'a'
    file_exists = os.path.exists(filename)

    with open(filename, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if overwrite or not file_exists:
            # Write header and data
            writer.writerows(data)
        else:
            # Append data (skip header)
            writer.writerows(data[1:])


# Grab already found posts
def get_ids(path=ID_LIST_PATH):
    if not os.path.exists(path):
        return set()

    values = set()
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            if row:
                values.add(row[0].strip())
    return values


# Grab already found posts in dataset
def get_used_ids(path=DATASET_PATH):
    if not os.path.exists(path):
        return set()

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    if "SubmissionID" not in df.columns:
        raise KeyError(f"'SubmissionID' column not found. Available columns: {list(df.columns)}")

    # Drop NaN and convert to strings
    return set(df["SubmissionID"].dropna().astype(str))


############################
# Data Scraping Functions
############################

# Grab posts from top, hot, new, rising, controversial
def pre_made_funcs(IDS, rows, subred, delay=DELAY):
    catalogues = {
        "top": "subred.top(limit=LIMIT)",
        "hot": "subred.hot(limit=LIMIT)",
        "new": "subred.new(limit=LIMIT)",
        "rising": "subred.rising(limit=LIMIT)",
        "controversial": "subred.controversial(limit=LIMIT)",
        "random_rising": "subred.random_rising(limit=LIMIT)"
    }

    for name, func in tqdm(catalogues.items(), desc="Checking categories", colour="green", total=len(catalogues)):
        total_len = len(list(eval(func)))
        counter = 0

        # Per-category counters (only count NEW unique posts for these)
        new_unique = 0
        new_gallery = 0
        new_inline = 0

        p_bar = tqdm(eval(func), desc=f"Finding {name} Posts: Found {counter} posts", colour="magenta", total=total_len, leave=False)
        for post in p_bar:
            # Check Uniqueness
            if post.id in IDS:
                continue
            new_unique += 1

            # Check if it has an image at all
            is_gallery = has_multiple_images(post, limit=1)
            is_inline = (not is_gallery) and has_at_least_one_image(post)

            if not is_gallery and not is_inline:
                continue
            if is_gallery:
                new_gallery += 1
            if is_inline:
                new_inline += 1

            IDS.add(post.id)
            counter += 1
            p_bar.set_description(f"Finding {name} Posts: Found {counter} posts")

        rows.append({
            "category": name,
            "num_posts_total": total_len,
            "num_new_unique": new_unique,
            "num_new_unique_gallery": new_gallery,
            "num_new_unique_inline": new_inline,
            "num_new_unique_with_image": new_gallery + new_inline,
        })

        time.sleep(delay)

    return IDS, rows


# Grab posts from searches (default is a-z)
def searches_for_posts(IDS, rows, subred, delay=DELAY, searches=None):
    func = f"subred.search(search,limit=LIMIT)"
    # counters
    new_unique = 0
    new_gallery = 0
    new_inline = 0
    new_posts = 0

    if not searches:
        searches = [chr(c) for c in range(ord('a'), ord('z') + 1)]

    for search in tqdm(searches, desc="Trying different searches", colour="green"):
        total_len = len(list(eval(func)))
        new_posts += total_len
        counter = 0

        p_bar = tqdm(eval(func), desc=f"Searching for {search} Posts: Found {counter} posts", colour="magenta",
                     total=total_len, leave=False)
        for post in p_bar:
            # Check Uniqueness
            if post.id in IDS:
                continue
            new_unique += 1

            # Check if it has an image at all
            is_gallery = has_multiple_images(post, limit=1)
            is_inline = (not is_gallery) and has_at_least_one_image(post)

            if not is_gallery and not is_inline:
                continue
            if is_gallery:
                new_gallery += 1
            if is_inline:
                new_inline += 1

            IDS.add(post.id)
            counter += 1
            p_bar.set_description(f"Searching for {search} Posts: Found {counter} posts")

        time.sleep(delay)

    rows.append({
        "category": f"searched {searches}",
        "num_posts_total": new_posts,
        "num_new_unique": new_unique,
        "num_new_unique_gallery": new_gallery,
        "num_new_unique_inline": new_inline,
        "num_new_unique_with_image": new_gallery + new_inline,
    })

    return IDS, rows


# Find Posts (Finds all potential IDs with images)
def find_posts(subreddit_name="AmIOverreacting", delay=DELAY):
    reddit = get_reddit_obj()
    subred = reddit.subreddit(subreddit_name)

    IDS = get_ids()
    rows = []

    # Run pre_made_funcs
    IDS, rows = pre_made_funcs(IDS, rows, subred, delay=delay)

    # Run searches_for_posts for each letter
    IDS, rows = searches_for_posts(IDS, rows, subred, delay=delay)

    SEARCHES = {
        "text",
        "screenshot",
        "image"
    }

    # Run searches_for_posts for each keyword
    IDS, rows = searches_for_posts(IDS, rows, subred, delay=delay, searches=SEARCHES)

    total_posts_sum = sum(row["num_posts_total"] for row in rows)
    new_unique_sum = sum(row["num_new_unique"] for row in rows)
    new_gallery_sum = sum(row["num_new_unique_gallery"] for row in rows)
    new_inline_sum = sum(row["num_new_unique_inline"] for row in rows)
    new_with_image_sum = sum(row["num_new_unique_with_image"] for row in rows)

    # Append the "Total" roll-up row
    rows.append({
        "category": "Total",
        "num_posts_total": total_posts_sum,
        "num_new_unique": new_unique_sum,
        "num_new_unique_gallery": new_gallery_sum,
        "num_new_unique_inline": new_inline_sum,
        "num_new_unique_with_image": new_with_image_sum,
    })

    # Write stats to JSON
    with open(f"{subreddit_name}_stats.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Write IDs to CSV
    with open(ID_LIST_PATH, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ids"])  # write header
        for val in IDS:
            writer.writerow([val])

    return IDS


###############################
# Generate Dataset
##############################3

# Download All images for a post
def get_images(submission):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent or "img-scraper/1.0"})

    try:
        sub_id = submission.id
        target_dir = os.path.join(IMAGE_DIR, sub_id)
        os.makedirs(target_dir, exist_ok=True)

        images = []

        # Check if gallery or single image
        if has_multiple_images(submission):
            # Gallery of Images
            for idx, item in enumerate(submission.gallery_data.get("items", []), start=1):
                media_id = item.get("media_id")
                md = submission.media_metadata.get(media_id, {}) or {}
                mime = md.get("m")
                url = None

                if isinstance(md.get("s"), dict):
                    url = md["s"].get("u") or md["s"].get("gif")
                if not url and isinstance(md.get("p"), list) and md["p"]:
                    url = md["p"][-1].get("u")
                if url:
                    images.append((idx, url.replace("&amp;", "&"), mime))
        else:
            # Single Inline Image
            url = (submission.url or "").replace("&amp;", "&")
            host = urlparse(url).netloc.lower()

            if "i.redd.it" in host:
                images.append((1, url, None))
            else:
                preview = getattr(submission, "preview", None)
                if isinstance(preview, dict):
                    im = (preview.get("images") or [])
                    if im and "source" in im[0] and "url" in im[0]["source"]:
                        url = im[0]["source"]["url"].replace("&amp;", "&")
                        images.append((1, url, None))

        # Download images and save in image directory
        for idx, url, mime in images:
            # Figure out extension
            ext = os.path.splitext(urlparse(url).path)[1].lower()
            if ext not in VALID_IMAGE_EXTS:
                ext = MIME_TO_EXT.get((mime or "").lower(), ".jpg")

            dest_path = os.path.join(target_dir, f"image{idx}{ext}")

            # Actual Download
            try:
                with session.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except Exception as e:
                print(f"[{sub_id}] failed #{idx} {url}: {e}")

    except Exception as e:
        print(f"{submission.id}: error -> {e}")

# Filter to see if Post has inline or preview images
def has_at_least_one_image(post):
    try:
        url = getattr(post, "url", "")
        image_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp")
        if any(url.lower().endswith(ext) for ext in image_exts):
            return 1

        if hasattr(post, "preview") and "images" in post.preview:
            return len(post.preview["images"])
    except Exception:
        return 0


# Checks if Post has gallery with multiple items
def has_multiple_images(post, limit=2):
    try:
        if getattr(post, "is_gallery", False):
            items = getattr(post, "gallery_data", {}).get("items", [])
            return len(items) >= limit
        else:
            return False
    except Exception:
        return False

# Get Comments for a submission by specified sort type
def get_comments_for_sort(reddit, submission_id, sort_name, max_comments=3, raise_errors=False):
    try:
        sub = reddit.submission(id=submission_id)
        sub.comment_sort = sort_name
        sub.comment_limit = 50

        # Fetch Comments to force PRAW to work
        sub._fetch()

        sub.comments.replace_more(limit=0)

        collected = []
        for c in sub.comments.list():
            if isinstance(c, Comment) and not getattr(c, "stickied", False):
                collected.append(c.body)
                if len(collected) >= max_comments:
                    break
        return collected

    except Exception as e:
        if raise_errors:
            raise
        return [f"Error fetching {sort_name} comments: {e}"]

# Generate the data set (Grabs all redddit data and Transcribes the images)
def generate_dataset(limit=-1, overwrite=False, safe=False):
    reddit = get_reddit_obj()

    IDS = get_ids()
    if not overwrite:
        # remove IDs that are already in the dataset
        IDS -= get_used_ids()

    if limit == -1:
        limit = len(IDS)

    header = DATASET_HEADER
    rows = [header]

    rejected_path = "RejectedIDS.csv"

    rejected_file_exists = os.path.exists(rejected_path)
    rejected_mode = "w" if overwrite or not rejected_file_exists else "a"

    with open(rejected_path, mode=rejected_mode, newline="", encoding="utf-8") as rejected_file:
        rejected_writer = csv.writer(rejected_file)
        if rejected_mode == "w":
            rejected_writer.writerow(["IDS"])  # header

        # Safe mode handling
        if safe:
            dataset_exists = os.path.exists(DATASET_PATH)
            if overwrite or not dataset_exists:
                with open(DATASET_PATH, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

        for index, id in tqdm(
            enumerate(list(IDS)[:limit], start=1),
            desc="Generating Dataset",
            colour="magenta",
            total=limit
        ):
            submission = reddit.submission(id=id)

            # Easy details
            url = submission.url
            title = submission.title
            body = submission.selftext

            # Submission-level stats
            score = getattr(submission, "score", None)
            upvote_ratio = getattr(submission, "upvote_ratio", None)

            # Comments for different sorts (up to 3 each), stored as JSON arrays
            best_comments = get_comments_for_sort(reddit, id, "confidence")
            top_comments = get_comments_for_sort(reddit, id, "top")
            controversial_comments = get_comments_for_sort(reddit, id, "controversial")
            qa_comments = get_comments_for_sort(reddit, id, "qa")

            best_comments_json = json.dumps(best_comments, ensure_ascii=False)
            top_comments_json = json.dumps(top_comments, ensure_ascii=False)
            controversial_comments_json = json.dumps(controversial_comments, ensure_ascii=False)
            qa_comments_json = json.dumps(qa_comments, ensure_ascii=False)

            # Transcription
            transcription = ""
            try:
                get_images(submission)
                print(f"\nTranscribing {id}\n")
                transcription = vision_transcribe.transcribe(id)

                if transcription is None:
                    transcription = "Error: Out of memory during vision transcription"
                else:
                    # Check for "rejected" transcription: {"messages": []}
                    try:
                        parsed = json.loads(transcription)
                        if (isinstance(parsed, dict) and isinstance(parsed.get("messages"), list) and len(parsed["messages"]) == 0):
                            rejected_writer.writerow([id])
                    except Exception:
                        pass

            except Exception as e:
                transcription = f"Error: {e}"

            row = (
                index,
                id,
                url,
                title,
                body,
                score,
                upvote_ratio,
                best_comments_json,
                top_comments_json,
                controversial_comments_json,
                qa_comments_json,
                transcription,
            )
            rows.append(row)

            if safe:
                # Append this single row immediately so progress is saved even if we crash
                with open(DATASET_PATH, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

    # In non-safe mode, write everything at once at the end
    if not safe:
        write_tuples_to_csv(rows, overwrite=overwrite)

    return rows

###########################
# Fix Dataset
###########################

# Regathers Reddit Data for an already generated dataset (As long as SubmissionID and Transcriptoin exist)
# Does not recalculate Transcriptions
def fix_dataset_from_file(fix_file_path):
    if not os.path.exists(fix_file_path):
        print(f"Error: fix file does not exist: {fix_file_path}")
        return

    # Load fix file
    fix_df = pd.read_csv(fix_file_path, dtype=str)

    # Check required columns
    required_cols = {"SubmissionID", "Transcription"}
    missing = required_cols - set(fix_df.columns)
    if missing:
        raise ValueError(f"Missing: {missing}. Found columns: {list(fix_df.columns)}")

    # Get Correct Headers
    dataset_df = pd.DataFrame(columns=DATASET_HEADER)
    reddit = get_reddit_obj()

    # Map SubmissionID -> existing row index
    subid_to_idx = {
        str(sid): idx
        for idx, sid in dataset_df["SubmissionID"].items()
    }

    updated = 0
    added = 0

    for _, row in tqdm(fix_df.iterrows(), total=len(fix_df), desc="Fixing dataset"):
        submission_id = str(row["SubmissionID"]).strip()
        if not submission_id:
            continue

        transcription_val = row["Transcription"]
        index_val = row["Index"] if "Index" in fix_df.columns else ""

        # Retry loop for this ID (rate limit handling)
        while True:
            try:
                # Fetch fresh Reddit data
                submission = reddit.submission(id=submission_id)

                url = submission.url
                title = submission.title
                body = submission.selftext
                score = getattr(submission, "score", None)
                upvote_ratio = getattr(submission, "upvote_ratio", None)

                # Fetch comments for each sort
                best_comments = get_comments_for_sort(reddit, submission_id, "confidence", raise_errors=True)
                top_comments = get_comments_for_sort(reddit, submission_id, "top", raise_errors=True)
                controversial_comments = get_comments_for_sort(reddit, submission_id, "controversial", raise_errors=True)
                qa_comments = get_comments_for_sort(reddit, submission_id, "qa", raise_errors=True)

                new_row = {
                    "Index": index_val,
                    "SubmissionID": submission_id,
                    "URL": url,
                    "Title": title,
                    "Body": body,
                    "Score": score,
                    "UpvoteRatio": upvote_ratio,
                    "BestComments": json.dumps(best_comments, ensure_ascii=False),
                    "TopComments": json.dumps(top_comments, ensure_ascii=False),
                    "ControversialComments": json.dumps(controversial_comments, ensure_ascii=False),
                    "QAComments": json.dumps(qa_comments, ensure_ascii=False),
                    "Transcription": transcription_val,
                }

                if submission_id in subid_to_idx:
                    idx = subid_to_idx[submission_id]
                    for col, val in new_row.items():
                        dataset_df.at[idx, col] = val
                    updated += 1
                else:
                    dataset_df = pd.concat(
                        [dataset_df, pd.DataFrame([new_row])],
                        ignore_index=True
                    )
                    # Update index mapping
                    subid_to_idx[submission_id] = dataset_df.index[-1]
                    added += 1

                break

            except Exception as e:
                print(f"Error processing {submission_id}: {e}")
                print("Sleeping 10 seconds, then retrying this ID...")
                time.sleep(10)

    dataset_df = dataset_df.reindex(columns=DATASET_HEADER)
    dataset_df.to_csv(fix_file_path, index=False, encoding="utf-8")

    print(f"Fix complete. Updated {updated} rows, added {added} new rows in {fix_file_path}.")




if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    # Search just generates a csv with potential SubmissionIDs
    group.add_argument(
        "--search",
        action="store_true",
    )

    # Generate takes the potential SubmissionIDs and generates the dataset
    group.add_argument(
        "--generate",
        nargs="?",
        const=-1,
        type=int,
    )

    # Fix takes the generated dataset and refreshes the reddit information without re-transcribing incase we changed the columns
    group.add_argument(
        "--fix",
        metavar="FILE",
    )

    args = parser.parse_args()

    if args.search:
        find_posts()
    elif args.generate is not None:
        overwrite = input("Overwrite existing dataset? (y/n): ").lower() == "y"

        # Safe mode inserts each row as it generates to preserve data in case it crashes
        safe = input("Run in safe mode? (y/n): ").lower() == "y"
        generate_dataset(args.generate, overwrite=overwrite, safe=safe)
    elif args.fix:
        fix_dataset_from_file(args.fix)
