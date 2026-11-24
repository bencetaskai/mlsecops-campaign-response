"""Values shared by the API tests."""

from pydantic import SecretStr

# Repeated characters: obviously fake and low in entropy, so gitleaks has nothing to flag and needs no exception.
API_KEY = SecretStr("k" * 40)
