from dagshub import get_repo_bucket_client


def upload_file_to_dagshub(repo_name, bucket_name, file_path, key):
    s3 = get_repo_bucket_client(repo_name, flavor="boto")
    s3.upload_file(
        Bucket=bucket_name,
        Filename=file_path,
        Key=key,
    )


def download_file_from_dagshub(repo_name, bucket_name, key, download_path):
    s3 = get_repo_bucket_client(repo_name, flavor="boto")
    s3.download_file(
        Bucket=bucket_name,
        Key=key,
        Filename=download_path,
    )


def delete_file_from_dagshub(repo_name, bucket_name, key):
    s3 = get_repo_bucket_client(repo_name, flavor="boto")
    s3.delete_object(
        Bucket=bucket_name,
        Key=key,
    )
