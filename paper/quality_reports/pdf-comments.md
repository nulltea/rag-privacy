# PDF comments — main.pdf (saved 2026-06-09 14:51)

## Detailed comments
 * Page #11 (Differential Privacy):
   > Within confidential inference it is a family rather than a single mechanism,

   This is redundant in the section intro. We should instead establish differential privacy in a unique way. Compared to static obfuscation and hybrid split schemes, the important fact to consider here is that some of the schemes from both of these other families do use differential privacy techniques internally, for example: AloePri uses Gaussian noise, Obfus LM also uses DP, and from the hybrid split site CSX uses Laplace.

   So we must establish the DP only category in terms of unique threat model and or performance/accuracy.

 * Page #11 (Differential Privacy): "Embedding-DP perturbs" -- I assume that embedding DP also requires client-side embedding, which is a deployment overhead access and routes some of the trust compute base into the client.

 * Page #11 (Differential Privacy): "provable bound" -- Provable how in a formal way? A separate question not to be included in the pros: How come the TP schemes have an approvable/formal definition of security, whereas obfuscation in a sense that these approaches apply mask does not? Does obfuscation share some of the mechanics with the information theoretic guarantees? In general, does the information theoretic argument apply to either DP or obfuscation?

 * Page #11 (Differential Privacy):
   > The outputside variant is notable for composing orthogonally with any retrieval mechanism,

   Output site variant here does it mean the generation using retrieved embedding/chunks as context? In other words, output site DP depends on the retrieval mechanism, or is it the other way around where output DP fits into the retrieval? I assume in this case it means that the output our embeddings. Need to clarify this.

 * Page #11 (Differential Privacy): "trading fidelity as ε tightens" -- The verb "tightens " here seems somewhat arbitrary. Wouldn't it better to use "increases"?

 * Page #11 (Approaches):
   > agree on the primitive, calibrated noise under a budget ε,

   What about approaches like Gaussian noise or Laplace (any other?) Do these underpin different types of DP noise?

   If yes, we need to report this distinction with trade-offs.

 * Page #11 (Approaches):
   > with a task-agnostic mechanism calibrated

   what does Task agnostic means it doesn't depend on a specific task? 

   Non-write Q: Is this calibration the reason why DP forward needs fine-tuning?

 * Page #11 (Approaches): "NVDP [16] replaces the fixed mechanism" -- What do you refer to when saying the fix mechanism—the transition from sparse to NV Reads inconsistent because Sparse also employs a learned approach? So, it's not clear and confusing what you counteract in NVTP when saying the fix mechanism vs the learned one.

 * Page #11 (Approaches): "stochastic posterior" -- Non-write Q: What is a posterior? Is this the standard term or an invented one?

 * Page #11 (Approaches): "shared representation" -- It's unclear what does a shared representation mean here and what's its relation to the DP noise?

 * Page #11 (Approaches): "sensitivity-selective shaping, or a learned task-aware model." -- What are the fundamental differences between these two? Because as far as I understand, the sensitive selective shaping you're referring to is also learned, as per NVDP.

 * Page #11 (Approaches):
   > Output perturbation. Two schemes bound what the generated answer reveals, again under distinct primitives.

   Non-write Q (to discuss): Does the input or embedding perturbation/noise carry not to the outputs in other way when embedding noise is applied to the inference? Do the resulting outputs come out perturbed or plain text? I assume that if they are not perturbed, that's a concrete requirement for having output perturbation, and if they do, the question is, is this inferior to targeted output perturbation or complementary, and how do they interact?

 * Page #11 (Approaches): "treats the retrieved corpus as the private dataset:" -- Retrieved corpus by means of what distance: cosine distance ranking or via inference? If it's the former and not inference, I assume that this scheme is better positioned into Section 8: Retrieval and Storage? to be discussed.

 * Page #11 (Approaches):
   > at each decode step it runs the model once on a public (sensitive-token-free) version of the context and once per sensitive group, then blends each private nexttoken distribution toward the public one until their Rényi divergence falls under the per-group budget, and samples from the mixture.

   Does this assume trusted hardware? Since, as you say, the scheme runs model separately per sensitive group, does it mean that the adversarial operator sees the sensitive information in plain text, or is it perturbed? Need to align description to be more informative of the security mechanism.

 * Page #11 (Approaches): "influence and, applied per retrieved chunk," -- What do you mean by retrieve chunk is the DP fusion approach targets? why do we mention chunks, does this scheme targets Decoder inference (generation) or re-ranker cross-encoder inference+ pooling?

 * Page #11 (Approaches): "are not interchangeable:" -- In other words, these two schemes are orthogonal or complementary; if they are better, say this instead of saying what they **are**, instead of what they're **not**

 * Page #11 (Approaches):
   > Query perturbation. RemoteRAG

   Query perturbation is specific to cosine similarity dense retrieval and not inference, I assume. In this case, it's probably better to drop remote rack from this section, Not to repeat what is already said in section 8.

 * Page #12 (Approaches): "6.2 Representative Schemes" -- Drop this section.

 * Page #12 (Representative Schemes): "6.3 Performance" -- We need to extend this section description to report not only on performance but accuracy as well, and to drive the point of the trade-off space between accuracy and security. In case performance is also affected by either of these, cover that too.

 * Page #12 (Performance (Reported)):
   > At inference every scheme here is near-plaintext in compute: the embedding schemes add one noise draw to the forward pass and the output schemes add token sampling. The real costs sit elsewhere and split the family. The embedding schemes carry an offline training bill,

   Does increasing the noise parameter or using a more secure noise metric technique carry the performance trade-off (Similar to how performance, security, and accuracy are interconnected with static obfuscation.)?

 * Page #12 (Performance (Reported)):
   > since DP-Forward fine-tunes (or pretrains) under the noise layer, SPARSE learns its per-concept mask, and NVDP trains the NVIB layer;

   Need to provide concrete numbers or the overhead, training overhead, scalability, slash complexity for each of those schemes to drive the point That overhead of certain DP schemes is pushed towards the offline phase..

 * Page #12 (Performance (Reported)):
   > DP-KSA multiplies generation by its ensemble size ( ≈N × calls, N ≈80), while DP-Fusion needs m+1 forward passes per token for m sensitive groups,

   These seem like genuine and noticeable overhead factors. We need to cover this in more detail (still concisely).

   And also include this into the performance table.

 * Page #12 (Performance (Reported)):
   > which parallelize so that latency stays near a single pass while compute and memory scale with m.

   Saying that parallelization incurs near-zero latency cost given that the memory and compute scales with M is a strong assumption about memory and compute being ample, In practice, both affect latency. 

   I assume that this parallelization also forces us to use specialized or bespoke inference software or orchestrate using the existing software? Which would the notable factor to report in terms of deployability overhead.

 * Page #12 (Performance (Reported)): "the shared currency," -- Currency here is a colloquial term. Use the standard contextually aligned and descriptive definition.

 * Page #12 (Performance (Reported)): "but degrading sharply as ε → 0 (" -- I think it's also important to report on The form of the slope at which increasing epsilon noise decreases accuracy: sublinear, linear, or superlinear? And whether it depends on the a) noise technique and/or b) ingestion point.

 * Page #12 (Approaches): "6.4 Limitations and Known Attacks" -- This section must clearly cover how concrete attacks with references/citations. Break the specific components of the specific schemes/techniques and whether they are mitigable or structural and cannot be defended in a purely DP manner.

 * Page #12 (Limitations and Known Attacks): "standard (ε, δ)-DP" -- I don't recall us discussing standard DP prior to this section—specifically, which schemes use it and how it is unique compared to the other approaches. All of this must be stated in the approaches subsection.

 * Page #12 (Limitations and Known Attacks): "in proportion to their distance," -- What does distance refer here? Is it the cosine distance? If so, where does this distinguishability versus distance dependence come from, and which attack exploited it?
