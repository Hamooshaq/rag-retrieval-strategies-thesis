# Arsitektur Pipeline Penelitian

## Tujuan

Pipeline menjalankan eksperimen closed-corpus RAG secara reproducible pada PDF dan PPT/PPTX. Alur utamanya:

`ingest -> parse -> normalize -> structure -> chunk -> embed -> retrieve -> route -> build context -> refuse/generate -> evaluate -> persist`

## Batas arsitektur

- Satu proses Python dan penyimpanan file lokal.
- Tidak ada microservice, message broker, vector database terdistribusi, dashboard, atau multi-agent.
- Semua konfigurasi berasal dari YAML.
- Exact cosine similarity dipakai agar hasil ranking dapat diaudit.
- Raw retrieval disimpan sebelum generasi.

## Struktur modul

```text
src/
  ingestion/    parser PDF dan PPTX
  structure/    parent, child, neighbor, dan locator
  chunking/     fixed-size dan structure-bounded child chunks
  retrieval/    frozen embedder dan exact cosine index
  routing/      predicted rule router dan oracle router
  generation/   prompt, token budget, local generator
  evaluation/   retrieval, router, refusal, answer, efficiency, statistik
  experiments/  orkestrasi B0-B4 dan CLI
  storage/      data model, manifest, JSONL, resume, provenance
```

## Data flow

1. `load_document_manifest()` memvalidasi document ID, lokasi file, format, license, dan SHA-256.
2. `parse_pdf()` atau `parse_pptx()` menghasilkan `ParsedBlock` dengan locator.
3. `build_structure()` membentuk parent per halaman PDF atau per slide PPTX, child terikat parent, dan neighbor berurutan.
4. `chunk_fixed()` membentuk unit B1 yang boleh melewati batas block tetapi tetap menyimpan seluruh locator.
5. `SentenceTransformerEmbedder` menghasilkan vector dari model dan revision yang dibekukan.
6. `ExactCosineIndex` menormalkan vector lalu melakukan exhaustive dot product.
7. Setiap query di-embed satu kali; vector dan biaya encoding yang sama digunakan untuk B1-B4 agar latency tidak dipengaruhi urutan baseline.
8. `ExperimentRunner` melakukan warm-up di luar metrik, memakai seeded cyclic
   counterbalancing, memilih B0-B4, memanggil router, menyusun context di bawah
   token budget, dan menerapkan threshold.
9. `TransformersGenerator` menjalankan model lokal; output refusal menggunakan token literal `INSUFFICIENT_EVIDENCE`.
10. `evaluate_record()` dan `build_reports()` menghitung metrik dari raw record dan gold annotation.
11. `JsonlResultStore` menulis record atomik, menolak duplicate key, dan mendukung resume.

## Definisi unit

- `locator`: `page:<n>` untuk PDF atau `slide:<n>` untuk PPTX.
- `parent`: satu halaman PDF atau satu slide PPTX.
- `child`: token window yang tidak pernah melintasi parent.
- `neighbor`: parent sebelum atau sesudah parent terpilih dalam dokumen yang sama.
- `gold evidence`: satu atau beberapa locator yang secara bersama cukup mendukung reference answer.

Gold evidence dapat memiliki alternative support sets yang sama-sama sah dan
harus dibekukan sebelum run. Retrieval metric bersifat locator-level: gain
nDCG biner diberikan satu kali ketika suatu hit pertama kali mencakup gold
locator baru. Source-location accuracy memakai locator yang benar-benar masuk
context, dan locator/unit ID hanya ditambahkan jika text evidence ikut termuat.

## Konfigurasi eksperimen

- B0: no retrieved context.
- B1: fixed-size chunks.
- B2: structure-aware child-only.
- B3: predicted routing.
- B4: gold-type oracle routing.

B2-B4 memakai child index yang sama. Untuk parent route, skor parent adalah skor child tertinggi yang memetakan ke parent. Multi-hop memilih parent berbeda sebelum context expansion.

## Reproducibility

Setiap run menyimpan:

- run ID dan configuration hash;
- seed;
- document hash dan license;
- model identifier, revision, dan local snapshot;
- library versions dan hardware;
- query, predicted/gold type, route;
- ranking, score, unit ID, locator, dan text;
- context, prompt, output;
- token count dan latency;
- posisi urutan baseline dan apakah generator benar-benar dipanggil;
- timestamp dan error.

## Error handling

- Dokumen kosong/rusak menghasilkan error dengan document ID.
- Slide tanpa teks dicatat dan dilewati; deck seluruhnya kosong ditolak.
- Parent tanpa child ditolak oleh validator.
- Unit tanpa locator ditolak.
- Duplicate document, question, unit, atau run key ditolak.
- Gold evidence kosong hanya diizinkan untuk `unanswerable`.
- Context builder mengemas hit berurutan, memverifikasi budget, dan tidak
  menyimpan locator/unit ID untuk hit yang tidak menyumbang text evidence.
- Model yang tidak tersedia menghasilkan `ModelUnavailableError`.
- Run parsial dapat dilanjutkan karena key yang sudah selesai dibaca dari JSONL.
