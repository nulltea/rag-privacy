# PDF comments — main.pdf (saved 2026-06-09 22:43)

## Detailed comments
 * Page #7 (Trusted Execution Environments):
   > That accelerator is now vendor-plural: a confidential NVIDIA GPU (an H100/B200-class GPU-TEE) or a confidential NPU (the Huawei Ascend line).

   Now vendor plural is you correcting your previous writing. The explicit rule was not to do this. Rephrase that we surfaced/cover two options instead.

 * Page #7 (Trusted Execution Environments): "the performance/trust baseline rather than systematize in depth:" -- Duplication about keeping it as a baseline was already said earlier in the same paragraph. Deduplicate based on prose information flow/arguments.

 * Page #7 (Trusted Execution Environments): "enclave construction" -- Enclave "construction" is arbitrary here.

 * Page #7 (Trusted Execution Environments):
   > That Tier-3 hardware requirement, not the performance, is what motivates the Tier-2 commodityGPU hybrid split (§ 7).

   Bad sentence structure.

 * Page #7 (Trusted Execution Environments): "The strongest adversary in our corpus appears here: Opal [31]" -- Ascend-CC is also a malicious host, is it not?

 * Page #7 (Approaches): "Four steps recur:" -- "Recur" here isn't ideal.

 * Page #7 (Approaches): "enclave's measurement" -- "Measurement" here is arbitrary.

 * Page #7 (Approaches):
   > and output handling, where plaintext logging, tracing, and telemetry outside the boundary are forbidden by policy.

   Why is this the "output" handling? Isn't this the general TEE limitations?

 * Page #8 (Approaches):
   > clave memory bandwidth and the absence of a GPU, so it is impractical for large models; it is the floor of the family, not a deployment target.

   Why is bandwidth, only memory availability also a significant limitation factor that dictates whether certain model sizes can even fit and run on a CPU enclave?

 * Page #8 (Approaches):
   > The confidential-GPU band runs the model on a The confidential GPU trusted accelerator.

   Redundant wording and duplication.

 * Page #8 (Approaches): "baseline the rest of the paper measures against." -- Drop the baseline notion from this section entirely

 * Page #8 (Approaches):
   > Opal [31] carries the strongest adversary in our corpus, a malicious host that may tamper with computation outside the enclave,

   How does it achieve that?

 * Page #8 (Approaches): "The confidential accelerator need not be an NVIDIA GPU." -- Drop this sentence.

 * Page #8 (Approaches):
   > Task and binary attestation block a malicious host from injecting rogue operators into the execution.

   This sentence is confusing.

 * Page #8 (Approaches):
   > it serves three mutually distrusting parties at once (the data owner, the model owner, and the cloud),

   What does it mean it serves?

 * Page #8 (Performance):
   > the confidential GPU adds 4–8% on Llama-2 7–70B, and Ascend-CC under 0.1% on Llama2/3 at realistic sequence lengths.

   This relative overhead number misses the concrete performance characteristics of the GPU enclave. NVIDIA is known to provide the top-of-the-line performance, whereas Ascend-CC I assume has the lower baseline performance (I assume it has lower transistor density + memory bandwidth).

 * Page #8 (Performance): "because the fixed trust cost amortizes over more computation;" -- Exactly why is it amortized?

 * Page #8 (Performance):
   > the percentage overheads that look large (Ascend-CC's 16% at a 50-token input) come only from tiny workloads where the trust machinery is not amortized.

   You're referencing this number as if It was introduced early, or readers have read the Ascend CC paper.

 * Page #8 (Performance): "on the current Hopper installed base" -- What is the Hopper installed base, and why do you refer to it as current?

 * Page #8 (Performance): "literature finds dominates" -- Finds to dominate?

 * Page #8 (Performance): "to hold a confidential GPU near 1.2× [65]." -- Near one point two x in relation to what? Is it more or less?

 * Page #8 (Performance): "training in simulation," -- Training in simulation? What does it mean?

 * Page #8 (Performance): "at the protocol level" -- "at the protocol" as if in software or in hardware?

 * Page #8 (Performance): "which the installed base does not yet [53]." -- What is the install base you're referring to?

 * Page #8 (Performance): "above the band:" -- "Band" here is arbitrary.

 * Page #8 (Performance):
   > still reports 29× lower infrastructure than a comparable secure baseline [31].

   What does lower infrastructure mean? Validate these numbers and make them relevant.

 * Page #8 (Limitations and Known Attacks):
   > The attacks here split by whether a scheme can close them, and the line between the two kinds is the section's main caveat (Table 5). The structural kind is not closed by anything in this family.

   The structural kinds are not closed by anything in this family, saying the same as the previous sentence.

 * Page #8 (Limitations and Known Attacks): "whether a scheme" -- Using a "scheme" universally in this section is not correct because many of the reported items are not schemes but rather hardware which has been evaluated.

 * Page #8 (Limitations and Known Attacks): "None is defended by the schemes here." -- Again, use of the word schemes. And second, more importantly, is that many if not all of these attacks are already patched in hardware (as far as I know, doublecheck). Find attacks that are structural to hardware and are impossible to fix (if any)

 * Page #8 (Limitations and Known Attacks):
   > We record this as the field-wide blind spot of § 9.5,

   Again, this is not a blind spot. This is a well-studied and There is a lot of hardware and security engineering to patch those vulnerabilities in hardware.

 * Page #8 (Limitations and Known Attacks):
   > which falls hardest on exactly the hardware-anchored families because their whole guarantee is the silicon.

   Need to explain why relying on silicon can be bad in the long term. Even if the current vulnerabilities are patched, there can be future ones and and meta question: why do they keep appearing?

 * Page #9 (Limitations and Known Attacks): "client relies on can be manufactured." -- Can be forged (more appropriate verb, no?)

 * Page #9 (Limitations and Known Attacks): "demand fresh quotes;" -- What does it mean to demand a fresh quote? Is it a colloquial or standard term?

 * Page #9 (Limitations and Known Attacks):
   > now vendor-plural (an NVIDIA GPU or a Huawei NPU) rather than a single-vendor lock,

   Again, this now vendor plural wording. Drop it.

 * Page #9 (Limitations and Known Attacks):
   > Its one residual risk is uniform across the band and independent of the scheme:

   You say one residual risk and then provide two.
