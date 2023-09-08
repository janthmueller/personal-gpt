import secret_keys
import os


def set_secret_keys():
    """
    Get all secret keys from secret_keys.py and set them as environment variables
    """
    keys = {
        item: secret_keys.__getattribute__(item)
        for item in dir(secret_keys)
        if not item.startswith("__")
    }

    for k in keys.keys():
        print(f"Setting {k} as environment variable.")
        os.environ[k] = keys[k]
