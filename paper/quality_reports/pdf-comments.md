# PDF comments — main.pdf (saved 2026-06-08 13:17)

## Detailed comments
 * Page #8 (Static Obfuscation): "so a white-box adversary can analyze it in advance." -- I think just saying that the white box adversary can analyze in advance misses a lot of details, namely that the adversary can proactively gather intermediary results—results of the computation—and use them to train the deobfuscate model, as well as the adversary knows the exact structure of the obfuscation algorithms and obfuscation parameters. We should find a concise statement which encapsulates this ability of the adversary. No need to go into many details because this will be described later on; just a concise short statement.

 * Page #8 (Static Obfuscation): "obfuscation leads on performance and fidelity," -- I wouldn't say that the static obfuscation is good for fidelity across the board.

 * Page #8 (Approaches):
   > and the observation surface is the obfuscated input/output (and, for KV-cache schemes, the externalized cache).

   The observation surface is not only input, outputs, and KV cache; it's everything that the obfuscation scheme loads to an untrusted accelerator. This can include offloaded attention, offloaded KV cache, matrix multiplications, etc.

 * Page #8 (Approaches):
   > larger h buys more heuristic hiding but raises the transform's condition number

   What does it mean transform condition number? Is this a reference for accuracy? This is arbitrary. Need to be concrete.

 * Page #8 (Approaches):
   > with the secret matrices held in a TEE for key custody only (no compute is offloaded to the enclave); it adds <0.45% to prefill and is lossless.

   This seems somewhat misleading that this scheme doesn't offload anything to enclave. Is this a specialized scheme that only targets KV cache and thus must be applied together with some other obfuscation? Is it a scheme like a lawyer pry, or is it a standalone scheme that builds the security argument? On the fact that securing KV cache is enough to break attackers' ability to recover inputs and outputs, or only inputs? Need to be specific here.

 * Page #8 (Approaches):
   > (EE) [8] swaps each layer's operators for transformed equivalents that preserve the linear maps

   What are layers operators exactly? Is this referring to linear operations or non-linear operations? Need to be specific.

 * Page #8 (Performance and Accuracy):
   > The covariant matrix schemes (AloePri, KV-Cloak) and EE need only a one-time weight transform

   Is EE not a covariant scheme? Why are a AloePri, and KV clock in round brackets, but EE outside?

 * Page #9 (Performance and Accuracy): "making each training step 1.6–3.4×" -- Before the number wrt Eguard was ~1.7, and now it's 1.6-3.4 What gives?

 * Page #9 (Performance and Accuracy): "(server-side hours)" -- What does "server side hours" means? This doesn't illustrate the argument that Obfus LM is heaviest. Did you not find the exact overhead or cost to fine-tune the model in the Obfus LM paper?

 * Page #9 (Performance and Accuracy): "algebraic identities)," -- Is "identities" a correct term here? Why not invariance or so? Double-check this.

 * Page #9 (Performance and Accuracy): "AloePri is the sharpest case—its keymat expansion h widens" -- Expansion factor H is not the only parameter that decreases fidelity for alloy; try the Gaussian and other—which I don't remember—contribute just as much or even more to accuracy variance. Double-check this with the documentation.

 * Page #9 (Performance and Accuracy):
   > The lesson for deployment is that the three axes are coupled: a scheme reporting ≈1× and near-lossless has typically spent its budget on the security axis, which is where the published attacks (§ 5.3) collect the bill.

   I want to emphasize insights like this by putting them into boxes (similar to how "Sok: Private Transformer-Based Model Inference" did it)

 * Page #9 (Limitations and Known Attacks):
   > Published attacks substantiate the field-level claim that static obfuscation under-protects relative to its reported guarantees.

   I don't want to be this aggressive and disputing the reported guarantees. Instead, I want to emphasize that finding a sweet spot in terms of accuracy, performance, and defense is possible but not trivial as it requires substantive ablation testing and tweaking, And parameters that work for a model with one set of hyperparameters won't work as effectively on a model with different hyperparameters or architecture. Thus portray this as a separate deployment difficulty factor. In a way, this becomes a tradeoff distinction between static obfuscation schemes that do not require training, but require careful choice of parameters from the schemes that avoid it by using trained networks or fine-tuning.

 * Page #9 (Limitations and Known Attacks): "glide-reflection-style" -- What is this term? Where is it coming from?

 * Page #9 (Limitations and Known Attacks): "solve for it" -- As far as I remember, this is called "ridge" attacks. Other attacks do not assume public weights but instead train a rich transformer using whatever offloaded externalities the deployment provides. Research this and ask me how to present this.

 * Page #10 (Limitations and Known Attacks):
   > The throughline across these results is a single design lesson: the schemes that survive are the ones whose hiding randomness is dynamic and never reused—KV-Cloak's per-block one-time permutation and AloePri's covariant transform on the static side, GELO's per-batch re-mixing on the hybrid side (§ 7).

   This argument is flawed. You're saying that schemes that survive use dynamic obfuscation, while inside a static obfuscation section, also referencing AloePri covariant transform, which is static, and Gelo, which is a hybrid scheme.
