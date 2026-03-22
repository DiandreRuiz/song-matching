from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    vector_store_path: str
    clap_model_id: str
    songs_dir_path: str
    

@lru_cache
def get_settings() -> Settings:
    return Settings()
