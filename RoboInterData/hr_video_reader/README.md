# Selectively Download the High-resolution Video of Droid

This README documents how to install **Google Cloud SDK** via an offline package on a cluster and use **gsutil** to browse and selectively download DROID Raw data from Google Cloud Storage (GCS).

---

## 1. Background

The cluster environment does not have `gsutil` installed by default, so `gs://gresearch/robotics/droid_raw` cannot be accessed directly.
The solution is to download and extract the `google-cloud-sdk` offline package, run the installation script to add `gsutil` to the environment variables, and then use `gsutil` for directory browsing and on-demand downloading.

---

## 2. Installing Google Cloud SDK (Offline Package)

Download the offline package on your local machine first, then upload it to the cluster:

`https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-462.0.0-linux-x86_64.tar.gz`

### 2.1 Extract the Archive

```bash
tar -xzf /path/to/google-cloud-sdk-462.0.0-linux-x86_64.tar.gz -C ~/.
```

The extracted directory will typically be:

```text
~/google-cloud-sdk/
```

### 2.2 Run the Installation Script (Updates ~/.bashrc)

```bash
cd ~/google-cloud-sdk/
conda activate your_env
./install.sh
```
Note: The `gsutil` command will only be available within the specified conda environment (e.g., `your_env`).

During installation, you will be prompted:

- `Modify profile to update your $PATH and enable shell command completion?`
  - It is recommended to enter `Y`
- `Enter a path to an rc file to update, or leave blank to use [~/.bashrc]:`
  - Press Enter to use the default `~/.bashrc`

The installer will back up and update `~/.bashrc` (e.g., creating `~/.bashrc.backup`).

### 2.3 Activate the Environment

Choose one of the following:

**Option 1: Reopen the shell (Recommended)**
```bash
exit
# Log in again or open a new terminal
```

**Option 2: Manually source the profile**
```bash
source ~/.bashrc
```

### 2.4 Verify gsutil is Available

```bash
which gsutil
gsutil version -l
gsutil ls gs://gresearch/robotics/droid_raw
```

---

## 3. Browsing and Selectively Downloading with gsutil

### 3.1 List Directories Layer by Layer (Explore Paths)

```bash
gsutil ls gs://gresearch/robotics/droid_raw
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/
```

### 3.2 List Files of a Specific Episode

```bash
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-07-19/Wed_Jul_19_20:59:01_2023/
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-07-19/Wed_Jul_19_20:59:01_2023/recordings/
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-07-19/Wed_Jul_19_20:59:01_2023/recordings/MP4/
```

### 3.3 Download a Single MP4

```bash
gsutil cp   gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-07-19/Wed_Jul_19_20:59:01_2023/recordings/MP4/27904255.mp4   ./27904255.mp4
```

### 3.4 Download the Entire MP4 Directory (Including Stereo)

> Note: When copying recursively, the destination must be an **existing directory** or `.`.

```bash
mkdir -p ./MP4_1
gsutil -m cp -r   gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-07-19/Wed_Jul_19_20:59:01_2023/recordings/MP4   ./MP4_1/
```

### 3.5 Download Only Non-Stereo MP4 Files (Recommended)

Approach: First use `gsutil ls` to list the files in the directory, filter out `-stereo.mp4` files, then download the rest.
Example (bash):

```bash
MP4_DIR="gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-07-19/Wed_Jul_19_20:59:01_2023/recordings/MP4/"
mkdir -p ./MP4_nostereo

gsutil ls "${MP4_DIR}" | grep -E '\.mp4$' | grep -v -- '-stereo\.mp4$' > mp4_list.txt
gsutil -m cp -I ./MP4_nostereo < mp4_list.txt
```

---

## 4. Common Errors and Fixes

### 4.1 `gsutil cp .../MP4 ./MP4_1` Shows "Did you mean to do cp -r?"

Cause: `.../MP4` is a directory prefix, not a file.
Solution: Use the `-r` flag for directories, or specify the exact file path (e.g., `.../MP4/27904255.mp4`).

### 4.2 `gsutil cp -r ... ./MP4_1` Reports "Destination URL must name a directory ..."

Cause: Recursive copy requires the destination to be a **directory**, and it should be created beforehand.
Solution:

```bash
mkdir -p ./MP4_1
gsutil cp -r gs://.../MP4 ./MP4_1/
# Or use the current directory as the destination
gsutil cp -r gs://.../MP4 .
```

---

## 5. Performance Tips

- For downloading large numbers of files, add the `-m` flag (parallel transfer):
  ```bash
  gsutil -m cp ...
  ```
- Avoid recursively syncing the entire bucket at once. Instead, navigate to the specific episode first and then download.
