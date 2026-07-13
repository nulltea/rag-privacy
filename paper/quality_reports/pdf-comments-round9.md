# PDF comments — main.pdf (saved 2026-06-10 21:21)

## Detailed comments
 * Page #20: "Onthatreadingthepresent-dayfrontierisheldbythehybrid" -- You just said that the headline is that no scheme satisfies all three parameters and the deployment. Is per application on needs, so it's not right to say that certain scheme families are different here.

 * Page #20: "Readacrossthefamilies,oneshaperecurs:eachmaxi-mizesatmosttwoofthethreedeployment-readiness" -- Duplication with the previous introduction. Remove the same text from the introduction.

 * Page #20: "toowningtheaccelera-torthatholdsit." -- Not owning, but availability.

 * Page #20: "Differentialprivacy(§6)attainsaformalthreatmodel" -- Probabilistic formal threat model.

 * Page #20: "↔Thecriterionafamilyconcedesisnotincidentalbutforcedbythebasisitrestson,whichiswhynosinglepointonthissurfacedominatestherest." -- Conclude that the choice of the scheme depends on the application needs: Latency, security budget, etc Need to think carefully what considerations dictate which seems to use.

 * Page #20: "Turningthegeometry" -- What geometry?? Arbitrary.

 * Page #20: "heldbythehybridsplit" -- Not by hybrid split because of its high serialization overhead; it's the TEE and static obfuscation. Hybrid split comes into relevance when the use case does not require high-end confidential GPU, for example, embedding or re-ranking, so that the operator needs no to pay for that enterprise expensive hardware and instead use more available CPU TEE and untrusted commodity GPUs.

 * Page #21: "alreadyownsone," -- People or companies usually don't own it.

 * Page #21: "thatmusthaveaprovableguaranteeandcanabsorbitstwo-to-threeorderoverhead." -- The question is whether such applications even exist. I think that the only justifiable use of cryptographic inference is collaborative private inference using MPC. For example, for medical analysis on disjoint institutionalized data. Worth researching this further and perhaps placing this into the cross-cutting finding.

 * Page #21: "Thisisthecon-creteformofourdeployment-readinessclaim,anditpartswayswiththetrust-minimizingTEE-to-FHEtra-jectoryofAndreolettietal.[5]:theleaderstodayarepresent-daysystems,notway-stationstowardacrypto-graphicendpoint." -- What do you mean by leaders and way stations?? Part ways? This section sounds completely arbitrary and meaningless.

 * Page #21: "Forembedding,wherethemodelissmallandtheprotectedassetistheinput,staticob-fuscationandlocalDParenear-plaintext,andcrypto-graphicencoders(SHAFT,Euston)arethechoiceonlywhenanexactandprovableguaranteeisrequired." -- Hybrid split also becomes relevant for this model architecture and use case.

 * Page #21: "deployableschemesaretheoblivi-ousones," -- "Oblivious ones" What does this even mean?

 * Page #21:
   > at-restchannelsthatdistance-preservingencryptionleavesopen(§8.3).

   Distance preserving encryption is highly relevant, particularly the Caprice scheme, because it achieves near plaintext speed while hiding embeddings, and the added DP noise eliminates back-to-text attacks. 

   In my opinion, this is one of the most promising schemes. But still depends on the use case whether some amount of comparability is problematic. Need to research this point in more detail.

 * Page #21: "ORAM(Com-pass)" -- You are forgetting about PacMann, which enables the same with cryptographic guarantees.

 * Page #21: "volumehidingfromencryptedmulti-maps(XorMM,FLASH),andthecheapestprotec-tionfromanonymization(PrivGemo,ARoG)ataheuris-ticandunquantifiedcost." -- You are not here; you are not giving any trade-offs, deployment readiness, and use case/application settings matching at all.

 * Page #21: "TheTier-1labelof§2.1markswhatmustbetrustednottoleak,notwheretheworkruns," -- The trusted compute base does say where the work runs. It just omits one part of the computation, usually the start, where schemes assume client work because it would be incorrect to say that the client's trusted compute base is client. This is just the artifact of our classification that we are correcting here, closing the gap. You need to define this clearly without making our classification look incomplete, and also keep this concise and straightforward.

 * Page #21: "readingdownthefamilies" -- This is weird wording, very confusing.

 * Page #21: "Staticweightobfuscation(AloePri,KV-Cloak,ObfusLM)hastheclienttransformtheweightsandholdthesecret,so" -- Covariant static obfuscation schemes

 * Page #21: "andDP-Forwardshowstheresultingtensiondirectly,sincepushingmorelayersontotheclientraisesutilitybutalsoitscomputeandstoragecost[17]." -- How exactly does the DP forward show this? Plus, in the immediately next sentence you say that DP forward can be fine-tuned by operator. Contradiction.

 * Page #21: "SPARSEtrainsitsmaskinminutes,andNVDPaddsasinglelayertoapretrainedencoder." -- There is no point in reiterating the performance of this offline setup because, as you said, it runs by the operator. So this is completely beside the point of this paragraph.

 * Page #21: "userinthelooponeverytoken," -- Is it really on every token? So as CSX basically replaces TEE with user?? Verify this claim. This seems incredibly inefficient because the communication over WAN would create huge latency.

 * Page #21: "andtheobliviousgraphschemeskeepapositionmapandrunamulti-roundcon-trollerontheclient" -- State which schemes require this.

 * Page #21: "TheTEEisthecounterpoint:" -- Not a counterpoint, but a workaround—trading the reliance on the client for the trust in TEE

 * Page #21: "coarse-grained." -- Is this the appropriate word here?

 * Page #21: "pure-TEE" -- Need to specify that pure confidential accelerator or GPU TEE.

 * Page #21: "con-vertsconfidentiality" -- Not converts but trades this performance and accuracy for

 * Page #21: "Aconfidentialacceleratorisavailabletodayonlyonhigh-endenterpriseparts,theH100/H200/B200classofGPUandtheHuaweiAscendNPU," -- Verify this claim: is this really only two platforms? I think there are more; it's just that we only report in these two.

 * Page #21: "nocheaperconfidentialpart" -- "part"??

 * Page #21: "right-sizeit." -- Right size - colloquial word, Use more straightforward and standard.

 * Page #22: "accumulateobservations" -- Accumulate externalities. or residual leakage.

 * Page #22: "hiddenquantitybecomesidentifiable." -- What's a hidden quantity? becomes identifiable through what needs to be concrete. I see you are saying this in later sentences, then use ":"

 * Page #22: "falltoprompt" -- "Fall to" is the wrong wording.

 * Page #22: "schemesthatstand" -- Stand where? this is colloquial!

 * Page #22: "aone-timekeypergeneratedtoken." -- Needs verification, as was said before.

 * Page #22: "Refreshingisnec-essarybutnotsufficient,sincethesecond-orderGramstructuresurvivesevenaper-batchrefreshunderorthog-onalmixing(§7)." -- Verify that refreshing secret doesn't defend hide gram structure. And what attack does this exploits?

 * Page #22: "Defendingamaliciousoperatoralmostalwaysbuysahardwaredependency." -- This is a confusing name. What does it mean buy a hardware dependency??

 * Page #22: "andfullmalicioussecuritycostsmorestill," -- Don't use "more still" — word in a more understandable way.

 * Page #22: "surveyedschemeadopts(§3)." -- Why do you refer to section 3 here?

 * Page #22: "withstandamaliciousoperator" -- What's the difference between full malicious security and defense against a malicious operator?

 * Page #22: "throughhardwareorheavyintegritymachinery:theconfidentialaccelerators(Nvidia-CC,Opal,Ascend-CC),TwinShield'sU-VerifychecksoverapossiblymaliciousGPU(§7)," -- Not only through hardware, Opal secures it cryptographically or information theoretically so as TwinShields with U-verify.

 * Page #22: "NoTier-1schemeonapurelycryptographic,obfuscation,orDPbasisde-fendsamaliciousoperatorwithoutoneofthese." -- MPC schemes can be maliciously secure if they use a message authentication code? Research into this with regards to our schemes: whether they do use one or can use one, or if they are structurally incompatible with this due to other reasons. I think there is a gap in our classification.

 * Page #22: "paidintrustedhardwareorinclient-sidecost." -- Definitely not only through trusted hardware.

 * Page #22: "Readingdowntheresidual-leakage" -- Stop using reading down. This is the AI voice tell!

 * Page #22: "power,andphysicalchannels" -- Power is a physical channel. Is it not?

 * Page #22: "andthegraphrowsaretheemptiestcellsofthecoveragemap(??)." -- Don't reference table.

 * Page #22: "Twocomponentshavenoprivatetreatmentatall:buildingthegraphindexconfi-dentially," -- Doesn't ARoG describe this? Also, you don't say that they don't have it at all. It's just that other papers, including us, do not cover this.

 * Page #22: "andgeneratingovertheretrievedprivatesub-graphratherthanmerelyfetchingit." -- Drop this graph RAG doesn't use GNN.

 * Page #22: "Thebreadthofretrievalcoveragesetagainsttheabsenceofthesetwoiswhytheconfidential-RAGfrontiersitsattheretrievallayer(§8.3),andwhytheemptiestcellsofthemaparethefield'sclearestopenproblems." -- This sentence is arbitrary and duplicating.
