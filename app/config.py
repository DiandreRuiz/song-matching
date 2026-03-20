from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    clap_model_id: str = "laion/clap-htsat-unfused"
    vector_store_path: str = "./data/vector_store"

@lru_cache
def get_settings() -> Settings:
    return Settings()
