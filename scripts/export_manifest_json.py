"""Export the verified YAML manifest for dependency-minimal document authoring."""
from pathlib import Path
from storage.io import atomic_write_json, load_yaml, sha256_file

root = Path(__file__).resolve().parents[2]
source = root / 'research-code/data/final_dataset/documents.yaml'
payload = load_yaml(source)
payload['source_sha256'] = sha256_file(source)
atomic_write_json(root / 'work/proposal/corpus_manifest.json', payload)
