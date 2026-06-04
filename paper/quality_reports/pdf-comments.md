# PDF comments — main.pdf (saved 2026-06-04 20:20)

## Detailed comments
 * Page #9 (Cryptographic Inference: MPC, FHE, and Secret Sharing): "3.1 Approach" -- I think we should split into 2-3 paragraphs: MPC, FHE

 * Page #9 (Approach): "3.2 Cryptographic Inference" -- There's no point in having this subsection because the whole section is about cryptographic inference. Let's split the content of this subsection between subsection 3.1 and the section three intro.

 * Page #15 (Confidential Retrieval and Storage):
   > 8.1 Approach

   There's no point of having this subsection approach because there are multiple approaches for different use cases, like dense retrieval and graph retrieval.

   We instead talk about approaches in dedicated subsections 8.2 and 8.3. Having a separate mini section approach would be redundant and cause duplication.

 * Page #15 (Approach):
   > 8.2 Dense Retrieval (Naive RAG) Over a flat or inverted-file (IVF) vector index, the question is how to rank the database by similarity to a query without revealing the query, the matches, or the stored vectors (Table 9).

   We need to clearly explain how and what this dense retireval leak exactly and why it's a problem

 * Page #15 (Dense Retrieval (Naive RAG)): "flat or inverted-file (IVF)" -- Is this flat or inverted file taxonomy relevant to privacy schemes being applied to dense readers? Retrieval, if not, remove it and only keep dense retrieval via proximate nearest neighbor ANN.

 * Page #15 (Dense Retrieval (Naive RAG)): "PHE rerank" -- PHE, CKKS, and HE scoring should have a separate paragraph to describe it. I assume this is a separate subclass of techniques.

 * Page #16 (Graph Retrieval and Traversal (Graph/LightRAG)):
   > Graph-RAG retrieves over a graph-structured index or knowledge graph: entity/relation ANN lookups, one-hop adjacency expansion, and source_id fan-out (the LightRAG surface). These ops leak differently from dense search—node access pattern and adjacency volume are the surfaces—so they need a different toolkit

   We need to clearly explain how and what this graph search specific surfaces leak exactly and why it's a problem

 * Page #16 (Graph Retrieval and Traversal (Graph/LightRAG)): "LightRAG" -- Need a citation for Light RAG.

 * Page #16 (Graph Retrieval and Traversal (Graph/LightRAG)): "XorMM [39] retrieve,gt volume-hiding EMM (SSE) n/r" -- We have a XOR MM implementation in the parent directory, so we can measure this. Add this as a to-do.

 * Page #16 (Graph Retrieval and Traversal (Graph/LightRAG)):
   > GORAM [13] PrivGemo [50] retrieve,gt retrieve,gt ego-query, billion-edge n/r — 3PC sqrt-ORAM (reserve) dual-tower KG anon (HMAC) no crypto cost exact quality-only

   We need to find latency and communication for these.

 * Page #17 (Graph Retrieval and Traversal (Graph/LightRAG)):
   > Anonymization. Where exact cryptography is too costly, PrivGemo [50] anonymizes the knowledgegraph view—a dual-tower design with HMAC session identifiers—so a remote LLM reasons over de-identified entities at no cryptographic cost, trading provable confidentiality for quality-preserving obfuscation (the hybrid/anonymization lineage of § 7)

   This is very vague, Need to clearly convey how the brief demo approach is relevant for security, during graph view search retrieval, etc., and making it private.

 * Page #17 (Graph Retrieval and Traversal (Graph/LightRAG)):
   > 8.4 Limitations and Known Attacks

   1. Needs to be more concrete about attack types, with concrete attack names, citations

   2. Do not mix dense and graph retrieval attacks.

   3. Clearly explain how each scheme holds against relevant attacks, how they deffent.

   4. Need a table
