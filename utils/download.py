import os
import requests
import hashlib

from tqdm import tqdm
def download(url, path=None, overwrite=False, sha1_hash=None):
    '''
    Download files from a given URL

    Parameters
    ==========
    url: str
        URL where file is located
    path: str, optional
        Destination path to store downloaded file. By default
        stores to the current location with same name
    overwrite: bool, optional
        whether to overwrite destination file if another file with same name
        exists at given location
    sha1_hash: str, optional
        Expected sha1 hash in hexadecimal digits (will ignore existing file when hash is specified but doesn't match)

    Returns
    =======
    str
        The file path of the downloaded file

    '''
    if path is None:
        fname = url.split('/')[-1]

    else:
        path = os.path.expanduser(path)
        if (os.path.isdir(path)):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print(f'Downloading {fname} from {url}...')
        r = requests.get(url, stream=True)
        if (r.status_code != 200):
            raise RuntimeError(f'Failed downloading url{url}')

        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))
        return fname


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash
