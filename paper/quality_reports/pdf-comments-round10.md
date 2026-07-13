# PDF comments — main.pdf (saved 2026-06-11 23:45)

## Detailed comments
 * Page #18 (Confidential Retrieval and Storage): "8.1 Confidential Similarity over a Hidden Index" -- When introdusing schemes in each of the subsection use bold scheme names

 * Page #18 (Confidential Similarity over a Hidden Index): "two-server secret sharing." -- You mean two-party secret sharing? client one of the parties?

 * Page #18 (Confidential Similarity over a Hidden Index): "coarseretrieves" -- What does this mean? Is this the term used in the paper?

 * Page #19 (Access-Pattern Hiding): "dense function" -- Don't use dense adjectives to function. This is confusing.

 * Page #19 (Access-Pattern Hiding): "the index most accurate for high-dimensional embeddings." -- What does this mean?

 * Page #19 (Access-Pattern Hiding):
   > Despite the "graph" in their construction they are denseretrieval schemes: each vertex is a database vector, not a knowledge-graph entity.

   This is confusing. Why are they graph-based? But you are saying that the retrieval is on database vectors and not entities. Does it mean that it works over embedded chunks and not embedded entities?

 * Page #19 (Access-Pattern Hiding): "ANN quality" -- What's the quality here? You mean accuracy or utility?

 * Page #19 (Access-Pattern Hiding): "−22% latency" -- what's a minus 22% is more or less overhead??

 * Page #19 (Access-Pattern Hiding): "TEE baseline" -- Drop baseline whenever talking about TEE!

 * Page #19 (Confidential Knowledge-Graph Construction): "The RAG-native systems" -- What do you mean by RAG native systems?

 * Page #19 (Confidential Knowledge-Graph Construction):
   > PrivGemo's local Hand holds the raw graph and builds a question-specific view on-device,

   Remove this from here.

 * Page #19 (Confidential Knowledge-Graph Construction): "and ARoG assumes a pre-built graph it only anonymizes." -- Remove.

 * Page #19 (Confidential Knowledge-Graph Construction):
   > Confidential construction from text on an untrusted server remains open (§ 9).

   Remove also

 * Page #19 (Adjacency Expansion and Graph Storage): "fans each" -- Is "fans" the proper verb here?

 * Page #19 (Adjacency Expansion and Graph Storage): "not by RAG-designed schemes." -- Remove

 * Page #19 (Adjacency Expansion and Graph Storage):
   > from near-optimal (FLASH, within a small constant of the volume-hiding lower bound) to ≈1.2–2.5× (XorMM);

   It should be the other way around, sorry, mamma is the earlier scheme. First you mention the less performant scheme, then the more performant.

 * Page #19 (Adjacency Expansion and Graph Storage):
   > FLASH's conjunctive variant adds a keyword-pair access pattern of its own,

   What does this mean?

 * Page #19 (Adjacency Expansion and Graph Storage):
   > PeGraph combines searchable encryption with secret sharing for ranked adjacency queries over million-entity graphs in under a second

   Secret sharing with whom? Is it a two-party scheme? with client?

 * Page #19 (Adjacency Expansion and Graph Storage): "initialization under 3 minutes);" -- What does initialization mean?

 * Page #19 (Adjacency Expansion and Graph Storage):
   > H2O2RAM supplies a doubly-oblivious RAM substrate inside a TEE beneath such a store.

   Supplies for what?

 * Page #20 (End-to-End Confidential KG-RAG): "ANN top-n," -- why top n and not top k? Paper uses N, or you invented it.

 * Page #20 (End-to-End Confidential KG-RAG): "Where even a TEE is unavailable," -- Where T is unavailable is a wrong premise, TEE is always available. It's a matter of choice. Start the sentence differently.

 * Page #20 (End-to-End Confidential KG-RAG): "+13 pp ANN quality." -- What does plus 13 mean? Does it make quality better?

 * Page #20 (End-to-End Confidential KG-RAG): "ARoG" -- no citation for this paper, needed

 * Page #20 (End-to-End Confidential KG-RAG): "KGQA" -- What is this?

 * Page #20 (End-to-End Confidential KG-RAG): "Hand" -- Double quotes

 * Page #20 (End-to-End Confidential KG-RAG): "Brain" -- double quotes

 * Page #20 (End-to-End Confidential KG-RAG): "(the anonymization lineage of § 7)," -- Why do you reference this here?

 * Page #20 (End-to-End Confidential KG-RAG):
   > which is why the confidential-RAG frontier sits at the retrieval functions above rather than at an integrated graph stack (§ 9).

   This is confusing. I don't understand what you're trying to say and why you reference section 9.

 * Page #20 (Limitations and Known Attacks): "and the attacks consume that leakage per function." -- Exploit that leakage.

 * Page #20 (Limitations and Known Attacks): "trade exactness" -- Trade utility

 * Page #20 (Limitations and Known Attacks): "on ciphertext" -- On ciphertexts. It doesn't make sense to talk about comparisons and mention only one ciphertext, I think.

 * Page #20 (Limitations and Known Attacks):
   > The property-preservingencryption family is invertible from ciphertext plus an auxiliary data distribution

   Confusing.

 * Page #20 (Limitations and Known Attacks):
   > Distance-revealing search also exposes k-nearestneighbour result geometry, which alone suffices to recover values [34];

   Is this a surface that Caprise doesn't defend?

 * Page #20 (Limitations and Known Attacks):
   > The homomorphic and secret-shared scoring schemes keep the values themselves encrypted and avoid this class entirely.

   Do they avoid vec to text attacks or the leakage through neighboring results geometry?

 * Page #21 (Limitations and Known Attacks):
   > These attacks fail against the oblivious schemes: PIR (Panther, PIR-RAG) genuinely closes the access-pattern channel, and Compass's ORAM hides which node is touched.

   Do these defend against volume attacks?

 * Page #21 (Limitations and Known Attacks):
   > The oblivious schemes, not the distance-preserving ones, are the deployable dense option here (§ 9).

   Stop referencing section 9. Every time you do it, it's super confusing.
