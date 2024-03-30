from spacy_llm.util import assemble

nlp = assemble("config.cfg")

doc = nlp(
    """inventive conceptions amount to creative ideas for designing devices that are both original and useful. The generation of inventive conceptions is a key element of the inventive process. However, neural mechanisms of the inventive process remain poorly understood. Here we employed functional feature association tasks and event-related functional magnetic resonance imaging (MRI) to investigate neural substrates for the generation of inventive conceptions. The functional MRI (fMRI) data revealed significant activations at Brodmann area (BA) 47 in the left inferior frontal gyrus and at BA 18 in the left lingual gyrus, when participants performed biological functional feature association tasks compared with non-biological functional feature association tasks. Our results suggest that the left inferior frontal gyrus (BA 47) is associated with novelty-based representations formed by the generation and selection of semantic relatedness, and the left lingual gyrus (BA 18) is involved in relevant visual imagery in processing of semantic relatedness. The findings might shed light on neural mechanisms underlying the inventive process."""
)

print([(ent.text, ent.label_) for ent in doc.ents])
