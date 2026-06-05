# PDF comments — main.pdf (saved 2026-06-05 22:46)

## Detailed comments
 * Page #8 (Static Obfuscation):
   > 5.1 Approaches

   Focusing on only data or covariant is too broad for the approaches section.

   I think we should instead briefly mention data only or covariant obfuscation in the intro to the static obfuscation section.

   In the Approaches subsection, the structure spine would instead need to be Mechanisms for each of the scheme, with each of the scheme name in bold (Similar to Approaches in hybrid split section.)

 * Page #8 (Approaches): "token substitution (SANTEXT/CUSTEXT" -- Drop this.

 * Page #8 (Approaches): "learned embedding transforms (SGT [49]," -- Current embedding transforms are by definition not data only.

 * Page #8 (Approaches): "5.2 Representative Schemes" -- This subsection should be subsumed into the Approaches subsection. Again, same structure as in hybrid split sect.)

 * Page #8 (Representative Schemes):
   > 5.3 Performance (Reported)

   This section must be much more nuanced because there are definitely certain trade-offs with performance.

   For example, in AloePri, there is an expansion dimension H which creates an extra dimension for heuristical hiding. And these dimensions create performance overhead. I'm sure other schemes might have similar mechanisms that affect performance, even if slightly.

   We also need a section about accuracy. We need to consider whether it would be a separate subsection or combine with performance into the performance and accuracy subsection. Or the accuracy trade-offs would instead be reported in the attacks subsection. 

   I think what we need to do is create a markdown document to track accuracy, performance, and security trade-offs for each of the schemes. This will give us a good dimensional understanding of how all of this is intertwined, and with that understanding, we can decide how best to report on it. There is already a document like this for AloePri; search for it in ../docs dir. e.g. docs/research/aloepri-h-beta-interaction-2026-05-27.md

 * Page #10 (Hybrid Split: TEE + Obfuscation):
   > Table 5: Attacks against static obfuscation. Data-only and static-permutation schemes are broken; the surviving designs refresh their hiding randomness

   This table definitely misses a lot of attacks. You can see the AloePri Markdown to get a list of attacks that each reports on.
