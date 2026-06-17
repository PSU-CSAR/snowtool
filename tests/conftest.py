import os

MOCK_REGION = 'us-east-1'


def set_fake_creds() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'  # noqa: S105
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'  # noqa: S105
    os.environ['AWS_SESSION_TOKEN'] = 'testing'  # noqa: S105
    os.environ['AWS_DEFAULT_REGION'] = MOCK_REGION
    os.environ['AWS_REGION'] = MOCK_REGION


def set_fake_settings() -> None:
    """Settings overrides for tests"""
    pass


set_fake_creds()
set_fake_settings()
