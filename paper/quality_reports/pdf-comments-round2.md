# PDF comments — main.pdf (saved 2026-06-04 13:33)

## Detailed comments
 * Page #8 (Cryptographic Approaches: MPC, FHE, and Secret Sharing): "we treat that sub-family briefly" -- we cover

 * Page #9 (Approach): "used by Transformer-FHE" -- what is Transfomer-FHE? Reads like name of the scheme. If it's not needto disambiguate

 * Page #9 (Approach):
   > leveled, supporting computation only up to a fixed multiplicative depth L before an expensive bootstrap, which again forces the non-linearities into low-degree polynomial approximations that consume the depth budget. In both families the non-linear operations are the bottleneck, and this is the structural reason secure inference sits one-to-three orders of magnitude above plaintext (§ 3.2).

   what about TFHE schemes with programmable bootstrapping - do we cover any one that use it?

 * Page #9 (Approach): "3.2 Cryptographic Inference" -- Missing important description about CryptoMoE contributing making MoE routing private - why it's important vs naive and how done.

 * Page #9 (Cryptographic Inference (Exemplars; Depth Deferred)): "(Exemplars; Depth Deferred)" -- remove

 * Page #9 (Cryptographic Inference (Exemplars; Depth Deferred)):
   > space—2PC, 2PC with an offline dealer, distributed-MPC, mixture-of-experts routing, and non-interactive FHE—

   Avoid using these long "..–...–..." this is not natural and hard to read!

 * Page #9 (Cryptographic Inference (Exemplars; Depth Deferred)): "even the fastest secure inference sits one-to-three orders of magnitude" -- repeates what was already said in prev section. Mention this once in this or previous section, pick one or reword

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)):
   > Table 4: Cryptographic-inference exemplars.

   How does Li et al. report performance/Fidelity 

   how do they normalize?

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)): "SHAFT [45] 2PC, BERT-base" -- BERT-base - num params?

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)): "MoE 6.9–16.4B" -- which model exactly? what's the active params count?

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)): "MPC+evaluators, ModernBERT" -- num params

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)): "≈10.7 s†" -- non-informative, do thet report num tokens used?

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)): "≈1.4 s†" -- non-informative, do thet report num tokens used?

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)): "<5 s/inf" -- non-informative, do thet report num tokens used?

 * Page #10 (Cryptographic Inference (Exemplars; Depth Deferred)):
   > 3.3 Cryptographic Retrieval and Storage

   This section must dive deeper into application of each of the scheme/sub family of schemes, how these schemes enable privacy, thread models. What's makes them efficient or slow.

   For graph related schemes we must clearly explain how each of the scheme is relevant for GraphRAG/LightRAG.

 * Page #10 (Cryptographic Retrieval and Storage): "IVF" -- what's IVF?

 * Page #10 (Cryptographic Retrieval and Storage): "Reported cost" -- what exactly do these measure - query?

 * Page #10 (Cryptographic Retrieval and Storage): "2-round SE + HE scoring" -- what's SE? footnote

 * Page #10 (Cryptographic Retrieval and Storage): "3–300× vs PRAG" -- extrapolate

 * Page #10 (Cryptographic Retrieval and Storage): "∼0 ms (deployed)" -- do they report vec/s

 * Page #10 (Cryptographic Retrieval and Storage):
   > Reported cost

   Fidelity and comms missing, separate columns

   reported time - what this measures, query, gt, or?

 * Page #10 (Cryptographic Retrieval and Storage): "SSE)" -- SSE? footnote

 * Page #10 (Cryptographic Retrieval and Storage): "billion-edge" -- ?

 * Page #11 (Cryptographic Retrieval and Storage): "we treat the attack" -- treat - arbitrary word

 * Page #11 (Cryptographic Retrieval and Storage): "3.4 Performance (Reported)" -- This section just repeates what was said in previous two - useless! Need to reconsider what. we report here or drop it.
