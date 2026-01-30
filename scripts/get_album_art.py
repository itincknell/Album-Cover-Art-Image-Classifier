import os
import numpy as np
import pandas as pd
import json
import re
import requests
import time
import threading
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Value

# -----------------------------------------------------------------------------
# MusicBrainz Cover Art Dataset Builder
#
# Goal
# ----
# Build a labeled dataset of album cover images (JPG) and metadata for a set of
# genres and decades. For each (genre, year) combination, the script queries a
# MusicBrainz PostgreSQL mirror for album release groups that:
#   - are tagged with the target genre
#   - have front cover art available via the Cover Art Archive
#   - have no secondary release-group type (to avoid compilations, live, etc.)
#
# It then constructs the cover-art URL, downloads the image to a genre folder,
# and saves a per-genre CSV linking each row to a local image filename.
#
# Configuration
# -------------
# Database connection is supplied via environment variables (with defaults):
#   MB_DB_HOST        (default: 0.tcp.ngrok.io)
#   MB_DB_PORT        (default: 15857)
#   MB_DB_USER        (default: musicbrainz)
#   MB_DB_PASSWORD    (default: musicbrainz)
#   MB_DB_NAME        (default: musicbrainz_db)
#
# Outputs
# -------
# - <cwd>/<genre>/...jpg            downloaded images
# - <cwd>/<genre>_df.csv            metadata + image filename + decade label
# ----------------------------------------------------------------------------

# Function to download images
def download_image(url, save_path):
    """
    Download a binary image payload from `url` and write it to `save_path`.

    Returns
    -------
    bool
        True if download/write succeeds; False on missing URL or any exception.

    Notes
    -----
    - This is a simple network helper; it does not enforce timeouts or status
      checks, and it treats any exception as a failure.
    """
    if url:
        try:
            img_data = requests.get(url).content
            with open(save_path, 'wb') as handler:
                handler.write(img_data)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    else:
        return False

# Function to clean a string for safe filenames
def clean_filename(name):
    """
    Replace filesystem-problematic characters with underscores.

    This helps create filenames that work across Windows/macOS/Linux.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# Save image with unique file name
def download_and_save_image(images_dir, index, album, pbar, success_count, fail_count):
    """
    Derive a human-readable filename from album artist/title, ensure uniqueness
    with the DataFrame index, then download the image if not already present.

    Parameters
    ----------
    images_dir : str
        Destination folder for images for a specific genre.
    index : int
        Row index from df.iterrows(); used to update df and disambiguate filenames.
    album : pandas.Series
        Row containing at least artist_name, release_group_name, and imUrl.
    pbar : tqdm.tqdm
        Shared progress bar (updated from multiple threads).
    success_count, fail_count : multiprocessing.Value
        Shared counters displayed in tqdm postfix.

    Returns
    -------
    tuple (index, file_name)
        Allows the caller to write `df.loc[index, 'image_file'] = file_name`.
    """
    # Create a stable filename that is both descriptive and collision-resistant.
    artist = clean_filename(album['artist_name'].replace(" ", "_")) if pd.notna(album['artist_name']) else "Unknown_Artist"
    title = clean_filename(album['release_group_name'].replace(" ", "_")) if pd.notna(album['release_group_name']) else "Unknown_Title"
    file_name = f"{artist}_{title}_{index}.jpg"
    save_path = os.path.join(images_dir, file_name)

    # If the image already exists, count it as a success and skip download.
    if os.path.exists(save_path):
        with pbar.get_lock():
            pbar.update(1)
            success_count.value += 1
            pbar.set_postfix(success=success_count.value, fail=fail_count.value)
        return index, file_name

    # Download image from the Cover Art Archive URL.
    success = download_image(album['imUrl'], save_path)

    # Update progress bar and shared counters safely across threads.
    with pbar.get_lock():
        pbar.update(1)
        if success:
            success_count.value += 1
        else:
            fail_count.value += 1
        pbar.set_postfix(success=success_count.value, fail=fail_count.value)

    return index, file_name

# Download images and save file names to df
def download_and_save_threaded(images_dir, df):
    """
    Concurrently download all images referenced by `df['imUrl']` and store the
    resulting local filename in `df['image_file']`.

    Concurrency model
    -----------------
    - Uses ThreadPoolExecutor, appropriate for I/O-bound work (HTTP downloads).
    - tqdm is updated from worker threads using its internal lock.
    """
    success_count = Value('i', 0)
    fail_count = Value('i', 0)

    with tqdm(total=len(df), desc="Downloading images", unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit one task per DataFrame row. Each task returns (index, file_name).
            futures = {
                executor.submit(download_and_save_image, images_dir, index, album, pbar, success_count, fail_count): index
                for index, album in df.iterrows()
            }

            # As tasks complete, write back the resolved filename to the DataFrame.
            for future in concurrent.futures.as_completed(futures):
                index, file_name = future.result()
                df.loc[index, 'image_file'] = file_name

    return df

# Ensure the images directory exists
images_dir = os.getcwd() + "/images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

from sqlalchemy import create_engine

# Database connection parameters via environment variables
# These defaults match a common MusicBrainz docker mirror setup, but can be
# overridden at runtime without editing the code.
ngrok_address = os.getenv("MB_DB_HOST", "0.tcp.ngrok.io")
ngrok_port = os.getenv("MB_DB_PORT", "15857")
username = os.getenv("MB_DB_USER", "musicbrainz")
password = os.getenv("MB_DB_PASSWORD", "musicbrainz")
database = os.getenv("MB_DB_NAME", "musicbrainz_db")

# SQLAlchemy connection string (psycopg2 driver)
connection_string = f"postgresql+psycopg2://{username}:{password}@{ngrok_address}:{ngrok_port}/{database}"
engine = create_engine(connection_string)

# year frequencies
def categorize_by_decade(year):
    """
    Map a release year (e.g., 1997) to a decade label (e.g., '1990s').

    Returns 'Unknown' when `year` cannot be cast to int.
    """
    try:
        return f"{int(year) // 10 * 10}s"
    except ValueError:
        return "Unknown"

# Function to create cover art URL
def create_cover_art_url(row, size='250'):
    """
    Construct a Cover Art Archive URL for a specific release.

    The Cover Art Archive uses the release MBID (release_id) and cover_art_id
    to reference a particular image variant. The size suffix selects a resized
    version (e.g., 250px).
    """
    if pd.notna(row['cover_art_id']):
        return f"http://coverartarchive.org/release/{row['release_id']}/{int(row['cover_art_id'])}-{size}.jpg"
    else:
        return None

def get_genre_df(genre, exclude_genres, limit, year, retries=3, delay=5):
    """
    Query MusicBrainz for album release groups in a specific year tagged with a
    given genre, including front cover art identifiers.

    Parameters
    ----------
    genre : str
        Target genre (must match genre.name in the database).
    exclude_genres : list[str]
        Other genres to exclude; helps keep the dataset more "single-genre".
    limit : int
        Maximum number of rows returned for the year.
    year : int
        Required match for first_release_date_year.
    retries : int
        Number of times to retry on query failure (e.g., transient network issues).
    delay : int
        Seconds to sleep between retries.

    Returns
    -------
    pandas.DataFrame
        One row per distinct release_group.name with relevant metadata and IDs.

    Filtering logic (SQL)
    ---------------------
    - cover_art.id IS NOT NULL: only include releases with cover art
    - cover_art_type.type_id = 1: restrict to front cover art
    - release_group.type = 1: restrict to album-type release groups
    - no secondary type: exclude compilations, live, etc. (as encoded in schema)
    - exclude any release group also tagged with other genres in `exclude_genres`
    """
    # PostgreSQL array literal, e.g. {"rock","pop"} for use with ANY(...)
    exclude_genres_str = "{" + ",".join([f'"{g}"' for g in exclude_genres]) + "}"

    # SQL is assembled as a string for pandas.read_sql. Inputs are controlled
    # by the script (not user input) in this workflow.
    query = f"""
    SELECT DISTINCT ON (release_group.name)
        release_group.gid AS release_group_mbid,
        release_group.name AS release_group_name,
        genre.name AS genre_name,
        release_group_meta.first_release_date_year AS release_year,
        artist.name AS artist_name,
        release.gid AS release_id,
        cover_art.id AS cover_art_id
    FROM
        release_group
    JOIN
        release_group_tag ON release_group.id = release_group_tag.release_group
    JOIN
        tag ON release_group_tag.tag = tag.id
    JOIN
        genre ON tag.name = genre.name
    JOIN
        release_group_meta ON release_group.id = release_group_meta.id
    JOIN
        artist_credit ON release_group.artist_credit = artist_credit.id
    JOIN
        artist_credit_name ON artist_credit.id = artist_credit_name.artist_credit
    JOIN
        artist ON artist_credit_name.artist = artist.id
    LEFT JOIN
        release ON release.release_group = release_group.id
    LEFT JOIN
        cover_art_archive.cover_art ON cover_art.release = release.id
    LEFT JOIN
        cover_art_archive.cover_art_type ON cover_art.id = cover_art_type.id
    LEFT JOIN
        release_group_secondary_type_join ON release_group.id = release_group_secondary_type_join.release_group
    WHERE
        genre.name = '{genre}'
        AND cover_art.id IS NOT NULL
        AND cover_art_type.type_id = 1
        AND release_group.type = 1  -- Assuming 1 represents 'album'
        AND release_group_secondary_type_join.release_group IS NULL  -- Ensures no secondary type
        AND release_group_meta.first_release_date_year = {year}
        AND release_group.gid NOT IN (
            SELECT release_group.gid
            FROM release_group
            JOIN release_group_tag ON release_group.id = release_group_tag.release_group
            JOIN tag ON release_group_tag.tag = tag.id
            JOIN genre ON tag.name = genre.name
            WHERE genre.name = ANY('{exclude_genres_str}')
        )
    LIMIT {limit};
    """
    
    # Retry loop is useful when tunneling (ngrok) or when DB is intermittently busy.
    for attempt in range(retries):
        try:
            start_time = time.time()
            genre_df = pd.read_sql(query, engine)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
            return genre_df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    raise Exception("Failed to execute query after multiple attempts")

def get_genre_dataset(genres, decades, size):
    """
    For each genre:
      - Create a genre-specific image folder in cwd.
      - Iterate each decade and year range.
      - Query the DB, generate URLs, label decades, download images, append to a
        per-genre DataFrame, then save to CSV.

    The `size` argument is treated as a target number of examples per genre.
    The script allocates size//10 examples per decade and tries each year in the
    decade; if a year comes up short, the deficit is carried to the next year
    within that decade.
    """
    for genre in genres:
        print(f"Downloading: {genre}")

        # Exclude other target genres to reduce multi-tag overlap.
        exclude_genres = [exc_genre for exc_genre in genres if exc_genre != genre]
        print(f"exclude_genres: {exclude_genres}")

        # Images are stored in a folder named after the genre.
        genre_dir = os.getcwd() + "/" + genre.replace(" ", "_")
        if not os.path.exists(genre_dir):
            os.makedirs(genre_dir)

        genre_df = pd.DataFrame()

        for decade in decades:
            print(f"Decade: {decade}")

            # Budget per decade and rolling deficit within the decade.
            limit = size // 10
            deficit = 0

            # Query year-by-year so the dataset spans the entire decade.
            for year in range(decades[decade][0], decades[decade][1]):
                print(f"Year: {year}")

                # If prior year did not meet its quota, request more this year.
                allowance = max(limit, limit + deficit)

                # Fetch rows for (genre, year).
                year_df = get_genre_df(genre, exclude_genres, allowance, year)

                if year_df.empty:
                    continue

                # Construct the Cover Art Archive URL used by the downloader.
                year_df['imUrl'] = year_df.apply(create_cover_art_url, axis=1)

                # Add a derived decade label used by downstream training code.
                year_df['decade'] = year_df['release_year'].apply(categorize_by_decade)

                # Extra dedupe to avoid repeated album titles within a year query.
                year_df.drop_duplicates(subset='release_group_name', inplace=True)

                print(f"{year_df.shape[0]} examples")

                # Carry deficit forward within the decade.
                deficit = allowance - year_df.shape[0]

                # Download images and record local filenames in `image_file`.
                year_df = download_and_save_threaded(genre_dir, year_df)

                # Append to the per-genre dataset.
                genre_df = pd.concat([genre_df, year_df])

        # Save per-genre metadata CSV alongside genre image folder.
        output_file = os.getcwd() + f'/{genre}_df.csv'
        genre_df.to_csv(output_file, index=False)

        # Summary counts by decade for quick sanity checks.
        genre_decade_counts = genre_df['decade'].value_counts().reset_index()
        genre_decade_counts.columns = ['decade', 'count']
        print(f"{genre} decade counts: {genre_decade_counts.sort_values('decade', ascending=False)}")

# Target genres
genres = ['rock', 'jazz', 'pop', 'classical', 'electronic']

# Decade ranges (start inclusive, end exclusive)
decades = {
    '1950s': (1950, 1960),
    '1960s': (1960, 1970),
    '1970s': (1970, 1980),
    '1980s': (1980, 1990),
    '1990s': (1990, 2000),
    '2000s': (2000, 2010),
    '2010s': (2010, 2020),
    '2020s': (2020, 2030)
}

# Target examples per genre (allocated across decades)
get_genre_dataset(genres, decades, 10_000)
