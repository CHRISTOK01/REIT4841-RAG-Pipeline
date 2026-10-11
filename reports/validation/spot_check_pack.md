# Manual spot-check pack

15 result rows from `results/full_sweep.jsonl`, sampled with seed 20261011 by `python -m scripts.build_spotcheck`.
The first 6 were selected because the automated scores disagree with each other; the other 9 are one random row per real configuration.

How to read a case: chunk labels are the pipeline's relevance labels (GOLD = cut from this query's own MedQuAD answer). Each claim line shows the judge's label against the retrieved context, then against the reference answer.

---

## Case 1: reduced / lexical

*Why this case is here:* **metrics disagree (A):** the gold chunk was retrieved at rank 1, yet faithfulness is low. (45 such rows in the sweep.)

**Query** (`GHR:0000688:0000688-5`): What are the treatments for multiple endocrine neoplasia ?

**Reference answer (MedQuAD; never shown to the generator):** These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Genetic Testing Registry: Familial medullary thyroid carcinoma - Genetic Testing Registry: Multiple endocrine neoplasia, type 1 - Genetic Testing Registry: Multiple endocrine neoplasia, type 2a - Genetic Testing Registry: Multiple endocrine neoplasia, type 2b - Genetic Testing Registry: Multiple endocrine neoplasia, type 4 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 1 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 2A - MedlinePlus Encyclopedia: Hyperparathyroidism - MedlinePlus Encyclopedia: Medullary Carcinoma of Thyroid - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) I - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) II - MedlinePlus Encyclopedia: Pancreatic Islet Cell Tumor - MedlinePlus Encyclopedia: Pheochromocytoma - MedlinePlus Encyclopedia: Pituitary Tumor - National Cancer Institute: Genetic Testing for Hereditary Cancer Syndromes - New York Thyroid Center: Medullary Thyroid Cancer T …[truncated]

**Relevant chunks in this KB:** 2 strict, 2 lenient.

**Retrieved context, as shown to the generator:**

1. `GHR:0000688:0000688-5#0` — label: **GOLD**; judge says passage is relevant; score 35.9171
   > These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Genetic Testing Registry: Familial medullary thyroid carcinoma - Genetic Testing Registry: Multiple endocrine neoplasia, type 1 - Genetic Testing Registry: Multiple endocrine neoplasia, type 2a - Genetic Testing Registry: Multiple endocrine neoplasia, type 2b - Genetic Testing Registry: Multiple endocrine neoplasia, type 4 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 1 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 2A - MedlinePlus Encyclopedia: Hyperparathyroidism - MedlinePlus Encyclopedia: Medullary Carcinoma of Thyroid - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) I - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia
2. `GHR:0000688:0000688-5#1` — label: **GOLD**; judge says passage is relevant; score 27.8584
   > MedlinePlus Encyclopedia: Medullary Carcinoma of Thyroid - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) I - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) II - MedlinePlus Encyclopedia: Pancreatic Islet Cell Tumor - MedlinePlus Encyclopedia: Pheochromocytoma - MedlinePlus Encyclopedia: Pituitary Tumor - National Cancer Institute: Genetic Testing for Hereditary Cancer Syndromes - New York Thyroid Center: Medullary Thyroid Cancer These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
3. `CancerGov:0000025_2:0000025_2-2#0` — label: **not relevant**; judge says passage is not relevant; score 21.1174
   > Having certain syndromes can increase the risk of pancreatic NETs. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk factors doesn't mean that you will not get cancer. Talk with your doctor if you think you may be at risk. Multiple endocrine neoplasia type 1 (MEN1) syndrome is a risk factor for pancreatic NETs.
4. `GARD:0004947:0004947-4#0` — label: **not relevant**; judge says passage is not relevant; score 15.4858
   > What treatments are available for itching related to polycythemia vera? There are several treatments for the itching (pruritus) related to polycythemia vera (PV). No single treatment has been found to be effective for all affected individuals. For mild cases, treatment may include avoiding triggers of itching and dry skin, or controlling the temperature of the environment and bathing water. Several other treatments are available for more severe itching or for itching the does not respond to initial treatments. Interferon-alpha has been found to be effective for reducing itching in a majority of individual with PV who received this therapy; however, this medication can have significant side effects. Selective serotonin reuptake inhibitors (SSRIs), typically used to treat depression, may reducing itching
5. `NINDS:0000202:0000202-4#1` — label: **not relevant**; judge says passage is not relevant; score 15.4845
   > In addition, there are a number of treatments under investigation that may curtail attacks or improve function. Over a dozen clinical trials testing potential therapies are underway, and additional new treatments are being devised and tested in animal models. In 2001, the National Academies/Institute of Medicine, a Federal technical and scientific advisory agency, prepared a strategic review of MS research. To read or download the National Academies/Institute of Medicine report, go to: "Multiple Sclerosis: Current Status and Strategies for the Future."

**Generated answer:**

```
## Summary
The context provided does not specify treatments for multiple endocrine neoplasia (MEN). It mentions resources related to the diagnosis and management of MEN but does not detail specific treatment options.

## Key Points
1. The resources listed address the diagnosis or management of multiple endocrine neoplasia but do not specify treatments [1][2].
2. There are various resources available for understanding MEN, including genetic testing and educational programs, but no treatments are mentioned [1][2].

## Disclaimer
This summary is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [no_information / no_information] The context provided does not specify treatments for multiple endocrine neoplasia (MEN).
- [unsupported / unsupported] It mentions resources related to the diagnosis and management of MEN but does not detail specific treatment options.
- [unsupported / unsupported] The resources listed address the diagnosis or management of multiple endocrine neoplasia but do not specify treatments.
- [unsupported / unsupported] There are various resources available for understanding MEN, including genetic testing and educational programs, but no treatments are mentioned.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.40 | 1.00 | 1.00 | 1.00 | 0.00 | abstained | undefined | 0.40 | 0.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 2: mixed / dense

*Why this case is here:* **metrics disagree (A):** the gold chunk was retrieved at rank 1, yet faithfulness is low. (45 such rows in the sweep.)

**Query** (`GHR:0000688:0000688-5`): What are the treatments for multiple endocrine neoplasia ?

**Reference answer (MedQuAD; never shown to the generator):** These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Genetic Testing Registry: Familial medullary thyroid carcinoma - Genetic Testing Registry: Multiple endocrine neoplasia, type 1 - Genetic Testing Registry: Multiple endocrine neoplasia, type 2a - Genetic Testing Registry: Multiple endocrine neoplasia, type 2b - Genetic Testing Registry: Multiple endocrine neoplasia, type 4 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 1 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 2A - MedlinePlus Encyclopedia: Hyperparathyroidism - MedlinePlus Encyclopedia: Medullary Carcinoma of Thyroid - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) I - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) II - MedlinePlus Encyclopedia: Pancreatic Islet Cell Tumor - MedlinePlus Encyclopedia: Pheochromocytoma - MedlinePlus Encyclopedia: Pituitary Tumor - National Cancer Institute: Genetic Testing for Hereditary Cancer Syndromes - New York Thyroid Center: Medullary Thyroid Cancer T …[truncated]

**Relevant chunks in this KB:** 2 strict, 2 lenient.

**Retrieved context, as shown to the generator:**

1. `GHR:0000688:0000688-5#0` — label: **GOLD**; judge says passage is relevant; score 0.6958
   > These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Genetic Testing Registry: Familial medullary thyroid carcinoma - Genetic Testing Registry: Multiple endocrine neoplasia, type 1 - Genetic Testing Registry: Multiple endocrine neoplasia, type 2a - Genetic Testing Registry: Multiple endocrine neoplasia, type 2b - Genetic Testing Registry: Multiple endocrine neoplasia, type 4 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 1 - Genomics Education Programme (UK): Multiple Endocrine Neoplasia type 2A - MedlinePlus Encyclopedia: Hyperparathyroidism - MedlinePlus Encyclopedia: Medullary Carcinoma of Thyroid - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) I - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia
2. `GHR:0000688:0000688-5#1` — label: **GOLD**; judge says passage is relevant; score 0.6108
   > MedlinePlus Encyclopedia: Medullary Carcinoma of Thyroid - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) I - MedlinePlus Encyclopedia: Multiple Endocrine Neoplasia (MEN) II - MedlinePlus Encyclopedia: Pancreatic Islet Cell Tumor - MedlinePlus Encyclopedia: Pheochromocytoma - MedlinePlus Encyclopedia: Pituitary Tumor - National Cancer Institute: Genetic Testing for Hereditary Cancer Syndromes - New York Thyroid Center: Medullary Thyroid Cancer These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
3. `CancerGov:0000027_1:0000027_1-9#29` — label: **not relevant**; judge says passage is not relevant; score 0.5642
   > The tumor is hormone receptor negative or positive. - The tumor is HER2/neu negative or positive. - The tumor is hormone receptor negative and HER2/neu negative (triple negative). - The size of the tumor. In premenopausal women with hormone receptor positive tumors, no more treatment may be needed or postoperative therapy may include: - Tamoxifen therapy with or without chemotherapy. - Tamoxifen therapy and treatment to stop or lessen how much estrogen is made by the ovaries. Drug therapy, surgery to remove the ovaries, or radiation therapy to the ovaries may be used. - Aromatase inhibitor therapy and treatment to stop or lessen how much estrogen is made by the ovaries. Drug therapy, surgery to remove the ovaries, or radiation
4. `CancerGov:0000003_5:0000003_5-8#31` — label: **not relevant**; judge says passage is not relevant; score 0.5524
   > by surgery. - Targeted therapy (mTOR inhibitor), for tumors that have certain gene changes and cannot be removed by surgery. - A clinical trial of targeted therapy, radiation therapy, and surgery with or without chemotherapy. Primitive neuroectodermal tumor (PNET)/extraskeletal Ewing tumor See the PDQ summary on Ewing Sarcoma Treatment. Synovial sarcoma Treatment of synovial sarcoma may include the following: - Chemotherapy. - Surgery. Radiation therapy and/or chemotherapy may be given before or after surgery. - A clinical trial of gene therapy. - A clinical trial of targeted therapy, radiation therapy, and surgery with or without chemotherapy. - A clinical trial of targeted therapy (histone methyltransferase inhibitor). - A clinical trial of an antibody-drug conjugate. Undifferentiated/unclassified sarcoma These tumors include undifferentiated
5. `CancerGov:0000003_5:0000003_5-8#26` — label: **not relevant**; judge says passage is not relevant; score 0.5416
   > Surgery and chemotherapy. - Radiation therapy. Malignant peripheral nerve sheath tumor Treatment of malignant peripheral nerve sheath tumor may include the following: - Surgery to completely remove the tumor when possible. - Radiation therapy before or after surgery. - Chemotherapy, for tumors that cannot be removed by surgery. - A clinical trial of targeted therapy, radiation therapy, and surgery with or without chemotherapy. - A clinical trial of targeted therapy, for tumors that cannot be removed by surgery. - A clinical trial of targeted therapy (histone methyltransferase inhibitor). - A clinical trial of an antibody-drug conjugate. It is not clear whether giving radiation therapy or chemotherapy after surgery improves the tumor's response to treatment. Malignant triton tumor Malignant triton tumors

**Generated answer:**

```
## Summary
The context provided does not specify the treatments for multiple endocrine neoplasia (MEN). It mentions resources related to the diagnosis and management of MEN but does not detail specific treatment options.

## Key Points
1. The context lists various resources that address the diagnosis or management of multiple endocrine neoplasia but does not provide specific treatment information [1][2].
2. No specific treatments for multiple endocrine neoplasia are mentioned in the provided context [1][2].

## Disclaimer
This summary is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [no_information / no_information] The context provided does not specify the treatments for multiple endocrine neoplasia (MEN).
- [unsupported / unsupported] It mentions resources related to the diagnosis and management of MEN but does not detail specific treatment options.
- [no_information / no_information] The context lists various resources that address the diagnosis or management of multiple endocrine neoplasia but does not provide specific treatment information.
- [no_information / no_information] No specific treatments for multiple endocrine neoplasia are mentioned in the provided context.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.40 | 1.00 | 1.00 | 1.00 | 0.00 | abstained | undefined | 0.40 | 0.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 3: reduced / lexical

*Why this case is here:* **metrics disagree (B):** nothing relevant was retrieved (P@k = 0), yet the answer does not abstain and is judged fully faithful. (71 such rows.)

**Query** (`MPlusHealthTopics:0000242:0000242-1`): What is (are) Cough ?

**Reference answer (MedQuAD; never shown to the generator):** Coughing is a reflex that keeps your throat and airways clear. Although it can be annoying, coughing helps your body heal or protect itself. Coughs can be either acute or chronic. Acute coughs begin suddenly and usually last no more than 2 to 3 weeks. Acute coughs are the kind you most often get with a cold, flu, or acute bronchitis. Chronic coughs last longer than 2 to 3 weeks. Causes of chronic cough include - Chronic bronchitis - Asthma - Allergies - COPD (chronic obstructive pulmonary disease) - GERD (gastroesophageal reflux disease) - Smoking - Throat disorders, such as croup in young children - Some medicines Water can help ease your cough - whether you drink it or add it to the air with a steamy shower or vaporizer. If you have a cold or the flu, antihistamines may work better than non-prescription cough medicines. Children under four should not have cough medicine. For children over four, use caution and read labels carefully.

**Relevant chunks in this KB:** 0 strict, 0 lenient.

**Retrieved context, as shown to the generator:**

1. `MPlusHealthTopics:0000213:0000213-1#0` — label: **not relevant**; judge says passage is relevant; score 14.2963
   > Summary : Sneezing, sore throat, a stuffy nose, coughing -- everyone knows the symptoms of the common cold. It is probably the most common illness. In the course of a year, people in the United States suffer 1 billion colds. What can you do for your cold or cough symptoms? Besides drinking plenty of fluids and getting plenty of rest, you may want to take medicines. There are lots of different cold and cough medicines, and they do different things. - Nasal decongestants - unclog a stuffy nose - Cough suppressants - quiet a cough - Expectorants - loosen mucus so you can cough it up - Antihistamines - stop runny noses and sneezing - Pain relievers - ease fever,
2. `MPlusHealthTopics:0000885:0000885-1#0` — label: **not relevant**; judge says passage is relevant; score 11.5232
   > Tetanus, diphtheria, and pertussis (whooping cough) are serious bacterial infections. Tetanus causes painful tightening of the muscles, usually all over the body. It can lead to "locking" of the jaw. Diphtheria usually affects the nose and throat. Whooping cough causes uncontrollable coughing. Vaccines can protect you from these diseases. In the U.S., there are four combination vaccines: - DTaP prevents all three diseases. It is for children younger than seven years old. - Tdap also prevents all three. It is for older children and adults. - DT prevents diphtheria and tetanus. It is for children younger than seven who cannot tolerate the pertussis vaccine. - Td prevents diphtheria and tetanus. It is for older children and adults. It is usually
3. `SeniorHealth:0000012:0000012-4#0` — label: **not relevant**; judge says passage is relevant; score 10.6014
   > Common Symptoms The most common symptoms of COPD are - a cough that does not go away - coughing up lots of sputum (mucus). a cough that does not go away coughing up lots of sputum (mucus). These symptoms often start years before the flow of air in and out of the lungs is reduced. Not everyone who has a cough and sputum goes on to develop COPD. Other common symptoms of COPD include - shortness of breath while doing activities you used to be able to do - wheezing (a whistling sound when you breathe) - tightness in the chest. shortness of breath while doing activities you used to be able to do wheezing (a whistling sound when you
4. `MPlusHealthTopics:0000213:0000213-1#1` — label: **not relevant**; judge says passage is not relevant; score 10.5828
   > so you can cough it up - Antihistamines - stop runny noses and sneezing - Pain relievers - ease fever, headaches, and minor aches and pains Here are some other things to keep in mind about cold and cough medicines. Read labels, because many cold and cough medicines contain the same active ingredients. Taking too much of certain pain relievers can lead to serious injury. Do not give cough medicines to children under four, and don't give aspirin to children. Finally, antibiotics won't help a cold. Food and Drug Administration
5. `MPlusHealthTopics:0000249:0000249-1#0` — label: **not relevant**; judge says passage is relevant; score 10.2921
   > Croup is an inflammation of the vocal cords (larynx) and windpipe (trachea). It causes difficulty breathing, a barking cough, and a hoarse voice. The cause is usually a virus, often parainfluenza virus. Other causes include allergies and reflux. Croup often starts out like a cold. But then the vocal cords and windpipe become swollen, causing the hoarseness and the cough. There may also be a fever and high-pitched noisy sounds when breathing. The symptoms are usually worse at night, and last for about three to five days. Children between the ages of 6 months and 3 years have the highest risk of getting croup. They may also have more severe symptoms. Croup is more common in the fall and winter.

**Generated answer:**

```
## Summary
Coughing is a common symptom associated with various illnesses, including the common cold and whooping cough. It can be a sign of respiratory issues and may involve coughing up mucus. Coughs can be treated with different types of medications, depending on their nature and underlying cause.

## Key Points
1. Coughing is a symptom of the common cold and can be treated with cough suppressants or expectorants, among other medications [1][4].
2. Whooping cough is characterized by uncontrollable coughing and is caused by a bacterial infection [2].
3. A persistent cough that produces mucus may indicate chronic obstructive pulmonary disease (COPD) [3].

## Disclaimer
This summary is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / unsupported] Coughing is a common symptom associated with various illnesses, including the common cold and whooping cough.
- [supported / unsupported] It can be a sign of respiratory issues and may involve coughing up mucus.
- [supported / supported] Coughs can be treated with different types of medications, depending on their nature and underlying cause.
- [supported / supported] Coughing is a symptom of the common cold and can be treated with cough suppressants or expectorants, among other medications.
- [supported / unsupported] Whooping cough is characterized by uncontrollable coughing and is caused by a bacterial infection.
- [supported / supported] A persistent cough that produces mucus may indicate chronic obstructive pulmonary disease (COPD).

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | undefined | undefined | undefined | 1.00 | answered | 0.50 | 0.80 | 0.50 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 4: curated / lexical

*Why this case is here:* **metrics disagree (B):** nothing relevant was retrieved (P@k = 0), yet the answer does not abstain and is judged fully faithful. (71 such rows.)

**Query** (`SeniorHealth:0000042:0000042-3`): What are the symptoms of Low Vision ?

**Reference answer (MedQuAD; never shown to the generator):** There are many signs that can signal vision loss. For example, even with your regular glasses, do you have difficulty - recognizing faces of friends and relatives? - doing things that require you to see well up close, such as reading, cooking, sewing, fixing things around the house, or picking out and matching the color of your clothes? - doing things at work or home because lights seem dimmer than they used to? - reading street and bus signs or the names of stores? recognizing faces of friends and relatives? doing things that require you to see well up close, such as reading, cooking, sewing, fixing things around the house, or picking out and matching the color of your clothes? doing things at work or home because lights seem dimmer than they used to? reading street and bus signs or the names of stores? Early Diagnosis Is Important Vision changes like these could be early warning signs of eye disease. People over age 60 should have an eye exam through dilated pupils at least once a year. Usually, the earlier your problem is diagnosed, the better your chances of undergoing successful treatment and keeping your remaining vision. Regular dilated eye exams should be part of your routi …[truncated]

**Relevant chunks in this KB:** 3 strict, 3 lenient.

**Retrieved context, as shown to the generator:**

1. `GARD:0003142:0003142-2#0` — label: **not relevant**; judge says passage is not relevant; score 19.9968
   > What are the signs and symptoms of idiopathic juxtafoveal retinal telangiectasia? Signs and symptoms of idiopathic juxtafoveal retinal telangiectasia may include slow loss of vision, distorted vision, trouble reading, and scotomata (a spot in the visual field in which vision is absent or deficient).
2. `SeniorHealth:0000042:0000042-1#0` — label: **not relevant**; judge says passage is relevant; score 18.2779
   > Everyday Tasks Are Challenging Low vision means that even with regular glasses, contact lenses, medicine, or surgery, people find everyday tasks difficult to do. Reading the mail, shopping, cooking, seeing the TV, and writing can seem challenging. Millions of Americans lose some of their vision every year. Irreversible vision loss is most common among people over age 65. (Watch the video to learn more about low vision. To enlarge the video, click the brackets in the lower right-hand corner. To reduce the video, press the Escape (Esc) button on your keyboard.) Not a Normal Part of Aging Losing vision is not just part of getting older. Some normal changes occur as we get older. However, these changes usually don't lead
3. `GHR:0000871:0000871-3#1` — label: **not relevant**; judge says passage is not relevant; score 18.0998
   > are thought to cause the X-linked form of the disorder. Together, mutations in the RPGR and RP2 genes account for most cases of X-linked retinitis pigmentosa. The genes associated with retinitis pigmentosa play essential roles in the structure and function of specialized light receptor cells (photoreceptors) in the retina. The retina contains two types of photoreceptors, rods and cones. Rods are responsible for vision in low light, while cones provide vision in bright light, including color vision. Mutations in any of the genes responsible for retinitis pigmentosa lead to a gradual loss of rods and cones in the retina. The progressive degeneration of these cells causes the characteristic pattern of vision loss that occurs in people with retinitis pigmentosa. Rods
4. `SeniorHealth:0000042:0000042-7#0` — label: **not relevant**; judge says passage is relevant; score 17.8139
   > Low vision can result from a variety of diseases, disorders, and injuries that affect the eye. Many people with low vision have age-related macular degeneration, cataracts, glaucoma, or diabetic retinopathy. Age-related macular degeneration accounts for almost 45 percent of all cases of low vision.
5. `GHR:0000871:0000871-3#2` — label: **not relevant**; judge says passage is not relevant; score 17.0540
   > progressive degeneration of these cells causes the characteristic pattern of vision loss that occurs in people with retinitis pigmentosa. Rods typically break down before cones, which is why night vision impairment is usually the first sign of the disorder. Daytime vision is disrupted later, as both rods and cones are lost. Some of the genes associated with retinitis pigmentosa are also associated with other eye diseases, including a condition called cone-rod dystrophy. Cone-rod dystrophy has signs and symptoms similar to those of retinitis pigmentosa. However, cone-rod dystrophy is characterized by deterioration of the cones first, followed by the rods, so daylight and color vision are affected before night vision.

**Generated answer:**

```
## Summary
Symptoms of low vision include difficulty performing everyday tasks such as reading, shopping, cooking, and watching TV, even with corrective lenses or surgery. It can also manifest as a slow loss of vision, distorted vision, and trouble reading [1][2][4].

## Key Points
1. Low vision makes everyday tasks challenging, even with regular glasses or surgery [2].
2. Symptoms may include slow loss of vision, distorted vision, and trouble reading [1][2].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / unsupported] Symptoms of low vision include difficulty performing everyday tasks such as reading, shopping, cooking, and watching TV, even with corrective lenses or surgery.
- [supported / unsupported] It can also manifest as a slow loss of vision, distorted vision, and trouble reading.
- [supported / supported] Low vision makes everyday tasks challenging, even with regular glasses or surgery.
- [supported / unsupported] Symptoms may include slow loss of vision, distorted vision, and trouble reading.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | answered | 0.75 | 0.40 | 0.25 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 5: reduced / lexical

*Why this case is here:* **metrics disagree (C):** every claim is judged supported by the retrieved context, but at most a third agree with the reference answer. (63 such rows.)

**Query** (`GARD:0004180:0004180-3`): What causes Multifocal motor neuropathy ?

**Reference answer (MedQuAD; never shown to the generator):** What causes multifocal motor neuropathy? The exact underlying cause of multifocal motor neuropathy (MMN) is poorly understood. It is considered an immune-mediated disorder (due to an abnormal immune system response), both because IVIG therapy improves symptoms, and many patients have anti-GM1 antibodies. Research to further understand the cause of MMN is underway.

**Relevant chunks in this KB:** 0 strict, 0 lenient.

**Retrieved context, as shown to the generator:**

1. `GHR:0000292:0000292-3#1` — label: **not relevant**; judge says passage is not relevant; score 11.8877
   > weakness in the hands and feet. The GARS gene provides instructions for making an enzyme called glycyl-tRNA synthetase, which is involved in the production (synthesis) of proteins. It is unclear how GARS gene mutations lead to distal hereditary motor neuropathy, type V. The mutations probably reduce the activity of glycyl-tRNA synthetase. A reduction in the activity of this enzyme may impair transmission of nerve impulses. As a result, nerve cells slowly lose the ability to communicate with muscles in the hands and feet. Mutations in other genes may also cause distal hereditary motor neuropathy, type V.
2. `GARD:0003371:0003371-1#0` — label: **not relevant**; judge says passage is not relevant; score 9.7936
   > What are the signs and symptoms of Keratoderma palmoplantar spastic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Keratoderma palmoplantar spastic paralysis. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Abnormality of the fingernails 90% Abnormality of the toenails 90% EMG abnormality 90% Gait disturbance 90% Muscle weakness 90% Palmoplantar keratoderma 90% Paresthesia 90% Pes cavus 90% Hemiplegia/hemiparesis 50% Hypertonia 50% Autosomal dominant inheritance - Heterogeneous - Motor axonal neuropathy - Nail dysplasia - Nail dystrophy -
3. `GARD:0003371:0003371-1#1` — label: **not relevant**; judge says passage is not relevant; score 9.5639
   > Hemiplegia/hemiparesis 50% Hypertonia 50% Autosomal dominant inheritance - Heterogeneous - Motor axonal neuropathy - Nail dysplasia - Nail dystrophy - Sensory axonal neuropathy - The Human Phenotype Ontology (HPO) has collected information on how often a sign or symptom occurs in a condition. Much of this information comes from Orphanet, a European rare disease database. The frequency of a sign or symptom is usually listed as a rough estimate of the percentage of patients who have that feature. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is the total number of people who were examined in one study. For example, a frequency
4. `GHR:0000292:0000292-3#0` — label: **not relevant**; judge says passage is relevant; score 9.1687
   > Mutations in the BSCL2 and GARS genes cause distal hereditary motor neuropathy, type V. The BSCL2 gene provides instructions for making a protein called seipin, whose function is unknown. Mutations in the BSCL2 gene likely alter the structure of seipin, causing it to fold into an incorrect 3-dimensional shape. Research findings indicate that misfolded seipin proteins accumulate in the endoplasmic reticulum, which is a structure inside the cell that is involved in protein processing and transport. This accumulation likely damages and kills motor neurons (specialized nerve cells in the brain and spinal cord that control muscle movement), leading to muscle weakness in the hands and feet. The GARS gene provides instructions for making an enzyme called glycyl-tRNA synthetase, which is
5. `GHR:0000077:0000077-3#1` — label: **not relevant**; judge says passage is not relevant; score 8.6397
   > both integral to the process of DNA replication by which new copies of mtDNA are produced. Mutated pol or mutated Twinkle reduce mtDNA replication. Although the mechanisms are unknown, mutations in the POLG gene often result in fewer copies of mtDNA (mtDNA depletion), and mutations in the C10orf2 gene often result in deletions of large regions of mtDNA (mtDNA deletion). MtDNA depletion or deletion occurs most commonly in muscle, brain, or liver cells. MtDNA depletion causes a decrease in cellular energy, which could account for the signs and symptoms of ataxia neuropathy spectrum. It is unclear what role mtDNA deletions play in the signs and symptoms of the condition.

**Generated answer:**

```
## Summary
The exact cause of Multifocal motor neuropathy is not detailed in the provided context passages. However, it is known that mutations in certain genes, such as the GARS and BSCL2 genes, can lead to related conditions like distal hereditary motor neuropathy, type V, which may share some characteristics with Multifocal motor neuropathy.

## Key Points
1. Mutations in the GARS gene can lead to distal hereditary motor neuropathy, type V, which involves reduced enzyme activity affecting nerve impulse transmission [1][4].
2. Mutations in the BSCL2 gene also contribute to motor neuropathy by causing misfolded proteins that damage motor neurons [4].

## Disclaimer
This summary is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [no_information / no_information] The exact cause of Multifocal motor neuropathy is not detailed in the provided context passages.
- [supported / unsupported] However, it is known that mutations in certain genes, such as the GARS and BSCL2 genes, can lead to related conditions like distal hereditary motor neuropathy, type V, which may share some characteristics with Multifocal motor neuropathy.
- [supported / unsupported] Mutations in the GARS gene can lead to distal hereditary motor neuropathy, type V, which involves reduced enzyme activity affecting nerve impulse transmission.
- [supported / unsupported] Mutations in the BSCL2 gene also contribute to motor neuropathy by causing misfolded proteins that damage motor neurons.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | undefined | undefined | undefined | 1.00 | partial | 0.68 | 0.20 | 0.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 6: reduced / lexical

*Why this case is here:* **metrics disagree (C):** every claim is judged supported by the retrieved context, but at most a third agree with the reference answer. (63 such rows.)

**Query** (`SeniorHealth:0000059:0000059-15`): What are the treatments for Psoriasis ?

**Reference answer (MedQuAD; never shown to the generator):** The goals of psoriasis treatment are to change the course of the disease by interfering with the increased production of skin cells, and to remove scales and smooth rough skin.

**Relevant chunks in this KB:** 1 strict, 1 lenient.

**Retrieved context, as shown to the generator:**

1. `SeniorHealth:0000059:0000059-7#0` — label: **not relevant**; judge says passage is relevant; score 22.2207
   > Scientists who are working to better understand and treat psoriasis are making headway in several different areas. The Role of T Cells Scientists believe that psoriasis occurs when white blood cells called T cells, which normally help fight infections, attack the bodys skin cells by mistake. Scientists are working to understand what causes these cells to go awry in people with psoriasis. Their hope is that by better understanding why T cells attack the bodys healthy skin tissue, they can develop better treatments to stop or prevent that damaging process. New Treatments Since discovering that T cells attack skin cells in psoriasis, researchers have been studying new treatments that quiet immune system reactions in the skin. Among these are treatments
2. `SeniorHealth:0000059:0000059-7#3` — label: **not relevant**; judge says passage is relevant; score 21.8747
   > to study the genetic aspects of psoriasis, and some studies are looking at the nervous system to determine the genes responsible for the circuitry that causes itching. Psoriasis-related Conditions Research in recent years has shown that people with psoriasis are more likely to develop other health problems, including problems with the heart and blood vessels. Research is continuing to examine links between psoriasis and other health problems. Scientists are working to understand how and why these diseases occur in people with psoriasis, with the hope that this understanding will lead to better treatments for both psoriasis and the related diseases. Stress Reduction Treatment For many people with psoriasis, life stresses cause the disease to worsen or become more active. Research
3. `SeniorHealth:0000059:0000059-7#1` — label: **not relevant**; judge says passage is relevant; score 21.1312
   > in psoriasis, researchers have been studying new treatments that quiet immune system reactions in the skin. Among these are treatments that block the activity of T cells or block cytokines (proteins that promote inflammation). If researchers find a way to target only the disease-causing immune reactions while leaving the rest of the immune system alone, resulting treatments could benefit psoriasis patients as well as those with other autoimmune diseases (when the immune system attacks the bodys own tissues). Currently there are a number of potential psoriasis treatments in clinical trials, including injections, pills, and topical ointments. Clinical trials are research studies with volunteers in which drugs are tested for the effectiveness and safety. All drugs must complete and pass this
4. `SeniorHealth:0000059:0000059-7#2` — label: **not relevant**; judge says passage is not relevant; score 17.5759
   > studies with volunteers in which drugs are tested for the effectiveness and safety. All drugs must complete and pass this process before they can be approved by the FDA. Psoriasis Genes Because psoriasis is more common among people who have one or more family members with the disease, scientists have long suspected that genes are involved. A number of genetic loci specific locations on the genes have been associated with the development of psoriasis or the severity or progression of the disease. In 2012, scientists discovered the first gene to be directly linked to development of plaque psoriasis. Researchers continue to study the genetic aspects of psoriasis, and some studies are looking at the nervous system to determine the genes
5. `GARD:0004947:0004947-4#0` — label: **not relevant**; judge says passage is not relevant; score 15.4858
   > What treatments are available for itching related to polycythemia vera? There are several treatments for the itching (pruritus) related to polycythemia vera (PV). No single treatment has been found to be effective for all affected individuals. For mild cases, treatment may include avoiding triggers of itching and dry skin, or controlling the temperature of the environment and bathing water. Several other treatments are available for more severe itching or for itching the does not respond to initial treatments. Interferon-alpha has been found to be effective for reducing itching in a majority of individual with PV who received this therapy; however, this medication can have significant side effects. Selective serotonin reuptake inhibitors (SSRIs), typically used to treat depression, may reducing itching

**Generated answer:**

```
## Summary
Treatments for psoriasis focus on quieting immune system reactions in the skin, particularly targeting T cells and cytokines that promote inflammation. Current potential treatments include injections, pills, and topical ointments, many of which are undergoing clinical trials to assess their effectiveness and safety.

## Key Points
1. Psoriasis treatments aim to quiet immune system reactions, specifically targeting T cells and cytokines that promote inflammation [1][3].
2. Potential treatments include injections, pills, and topical ointments, which are currently in clinical trials [3][4].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / unsupported] Treatments for psoriasis focus on quieting immune system reactions in the skin, particularly targeting T cells and cytokines that promote inflammation.
- [supported / unsupported] Current potential treatments include injections, pills, and topical ointments, many of which are undergoing clinical trials to assess their effectiveness and safety.
- [supported / supported] Psoriasis treatments aim to quiet immune system reactions, specifically targeting T cells and cytokines that promote inflammation.
- [supported / unsupported] Potential treatments include injections, pills, and topical ointments, which are currently in clinical trials.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | answered | 0.71 | 0.60 | 0.25 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 7: curated / lexical

*Why this case is here:* random draw for this configuration.

**Query** (`GARD:0002912:0002912-1`): What are the symptoms of Holzgreve syndrome ?

**Reference answer (MedQuAD; never shown to the generator):** What are the signs and symptoms of Holzgreve syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Holzgreve syndrome. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Aplasia/Hypoplasia of the lungs 90% Cleft palate 90% Hand polydactyly 90% Intrauterine growth retardation 90% Oligohydramnios 90% Renal hypoplasia/aplasia 90% Abnormal vertebral ossification 50% Abnormality of calvarial morphology 50% Abnormality of the mesentery 50% Abnormality of the metacarpal bones 50% Abnormality of the ribs 50% Abnormality of the ulna 50% Aplasia/Hypoplasia of the corpus callosum 50% Aplasia/Hypoplasia of the tongue 50% Bifid tongue 50% Limitation of joint mobility 50% Low-set, posteriorly rotated ears 50% Macrotia 50% Single umbilical artery 50% Webbed neck 50% Autosomal recessive inheritance - Cleft upper lip - Hypoplastic left heart - Renal agenesis - Renal hypoplasia - The Human Phenotype Ontology (HPO) has collected inf …[truncated]

**Relevant chunks in this KB:** 4 strict, 8 lenient.

**Retrieved context, as shown to the generator:**

1. `GARD:0002912:0002912-1#0` — label: **GOLD**; judge says passage is relevant; score 24.7712
   > What are the signs and symptoms of Holzgreve syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Holzgreve syndrome. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Aplasia/Hypoplasia of the lungs 90% Cleft palate 90% Hand polydactyly 90% Intrauterine growth retardation 90% Oligohydramnios 90% Renal hypoplasia/aplasia 90% Abnormal vertebral ossification 50% Abnormality of calvarial morphology 50% Abnormality of the mesentery 50% Abnormality of the metacarpal bones 50% Abnormality of the ribs 50% Abnormality of the ulna 50% Aplasia/Hypoplasia
2. `GARD:0002724:0002724-2#0` — label: **not relevant**; judge says passage is not relevant; score 15.4493
   > What are the signs and symptoms of Hanhart syndrome? The signs and symptoms of Hanhart syndrome vary, but may include: Small mouth Short, incompletely developed tongue (hypoglossia) Absent, partially missing, or shortened fingers and/or toes Jaw abnormalities such as micrognathia, retrognathia (receding jaw), or partially missing mandible (lower jaw) High-arched, narrow, or cleft palate Absent or unusually formed arms and/or legs Missing teeth Absence of major salivary glands Some infants with Hanhart syndrome may be born with paralysis of certain areas of the face. If the tongue and/or mouth are affected, this can worsen feeding difficulties that are already present due to the craniofacial abnormalities listed above. The severity of the physical abnormalities associated with Hanhart syndrome varies greatly among
3. `GARD:0006210:0006210-2#0` — label: **not relevant**; judge says passage is not relevant; score 14.9603
   > What are the signs and symptoms of Trismus-pseudocamptodactyly syndrome? While the symptoms of trismus-pseudocamptodactyly syndrome vary from patient to patient, characteristic symptoms include the inability to open the mouth wide (e.g., less than 6 mm, just under 1/4 of an inch) and shortened muscles, including of the hamstrings and calf muscles. As a result of shortened muscles some infants with trismus-pseudocamptodactyly syndrome have closed or clinched fists, club foot, metatarsus adductus, and calcaneovalgus (where the foot bends sharply at the ankle) at birth. Children with this syndrome may crawl on their knuckles. In adulthood the syndrome may cause reduced hand dexterity, however hand limitation does not often interfere with normal function. The most serious complications of the condition occur as
4. `GARD:0001664:0001664-3#0` — label: **not relevant**; judge says passage is not relevant; score 14.8907
   > What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals but studies suggest that larger deletions tend to result in more severe intellectual disability and developmental delay than smaller deletions. The signs and symptoms of cri du chat syndrome are probably related to the loss of multiple genes on the short arm of chromosome 5. Researchers believe that the loss of a specific gene, CTNND2, is associated with severe intellectual disability in some people with this condition. They are working to determine how the loss of other genes
5. `GARD:0006224:0006224-2#0` — label: **not relevant**; judge says passage is not relevant; score 14.8087
   > What are the signs and symptoms of Troyer syndrome? The signs and symptoms of Troyer syndrome can vary, and some people are more severely affected than others. Symptoms typically begin in early childhood. Most affected children have delays in walking and talking, followed by slow deterioration in both manner of walking (gait) and speech. Affected people have progressive muscle weakness and stiffness (spasticity) in the legs; muscle wasting in the hands and feet; paraplegia; leg contractures; learning disorders; and short stature. Mood swings and mood disorders, causing inappropriate euphoria and/or crying, are common. Other features can include drooling; exaggerated reflexes (hyperreflexia) in the legs; uncontrollable movements of the arms and legs (choreoathetosis); skeletal abnormalities; and a bending outward (valgus) of

**Generated answer:**

```
## Summary
Holzgreve syndrome is characterized by several symptoms, including aplasia or hypoplasia of the lungs, cleft palate, hand polydactyly, intrauterine growth retardation, oligohydramnios, and renal hypoplasia or aplasia, with each of these symptoms occurring in approximately 90% of affected individuals. Other symptoms, such as abnormal vertebral ossification, abnormalities of calvarial morphology, mesentery, metacarpal bones, ribs, and ulna, occur in about 50% of cases.

## Key Points
1. Symptoms of Holzgreve syndrome include aplasia/hypoplasia of the lungs, cleft palate, hand polydactyly, intrauterine growth retardation, oligohydramnios, and renal hypoplasia/aplasia, all seen in approximately 90% of patients [1].
2. Other symptoms, such as abnormal vertebral ossification and various skeletal abnormalities, are present in about 50% of individuals with Holzgreve syndrome [1].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / supported] Holzgreve syndrome is characterized by several symptoms, including aplasia or hypoplasia of the lungs, cleft palate, hand polydactyly, intrauterine growth retardation, oligohydramnios, and renal hypoplasia or aplasia, with each of these symptoms occurring in approximately 90% of affected individuals.
- [supported / supported] Other symptoms, such as abnormal vertebral ossification, abnormalities of calvarial morphology, mesentery, metacarpal bones, ribs, and ulna, occur in about 50% of cases.
- [supported / supported] Symptoms of Holzgreve syndrome include aplasia/hypoplasia of the lungs, cleft palate, hand polydactyly, intrauterine growth retardation, oligohydramnios, and renal hypoplasia/aplasia, all seen in approximately 90% of patients.
- [supported / supported] Other symptoms, such as abnormal vertebral ossification and various skeletal abnormalities, are present in about 50% of individuals with Holzgreve syndrome.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.25 | 1.00 | 0.39 | 1.00 | answered | 0.88 | 0.20 | 1.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 8: curated / dense

*Why this case is here:* random draw for this configuration.

**Query** (`NIDDK:0000094:0000094-11`): How to prevent Cyclic Vomiting Syndrome ?

**Reference answer (MedQuAD; never shown to the generator):** A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - avoid foods that trigger episodes or foods with additives A health care provider may refer people with cyclic vomiting syndrome and anxiety to a stress management specialist for relaxation therapy or other treatments. A health care provider may prescribe medications to prevent migraines for people with cyclic vomiting syndrome.

**Relevant chunks in this KB:** 1 strict, 1 lenient.

**Retrieved context, as shown to the generator:**

1. `NIDDK:0000094:0000094-11#0` — label: **GOLD**; judge says passage is relevant; score 0.6600
   > A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - avoid foods that trigger episodes or foods with additives A health care provider may refer people with cyclic vomiting syndrome and anxiety to a stress management specialist for relaxation therapy or other treatments. A health care provider may prescribe medications to prevent migraines for people with cyclic vomiting syndrome.
2. `NIDDK:0000098:0000098-7#0` — label: **not relevant**; judge says passage is not relevant; score 0.4559
   > Treatment for dumping syndrome includes changes in eating, diet, and nutrition; medication; and, in some cases, surgery. Many people with dumping syndrome have mild symptoms that improve over time with simple dietary changes.
3. `CDC:0000120:0000120-7#0` — label: **not relevant**; judge says passage is not relevant; score 0.4526
   > On the basis of the currently available information, avoiding food or water that may have been contaminated with feces is the best way to prevent cyclosporiasis. Treatment with chlorine or iodine is unlikely to kill Cyclospora oocysts. No vaccine for cyclosporiasis is available. The U.S. Food and Drug Administration's (FDA) Center for Food Safety and Applied Nutrition (CFSAN) publishes detailed food safety recommendations for growers and suppliers. In its Guide to Minimize Microbial Food Safety Hazards for Fresh Fruits and Vegetables, CFSAN describes good agricultural practices (GAPs) and good manufacturing practices (GMPs) for fresh fruits and vegetables. The guidelines address the growing, harvesting, sorting, packaging, and storage processes; following the guidelines can help reduce the overall risk for microbial contamination
4. `NINDS:0000169:0000169-2#0` — label: **not relevant**; judge says passage is not relevant; score 0.4345
   > There is no definitive treatment for Kleine-Levin syndrome and watchful waiting at home, rather than pharmacotherapy, is most often advised. Stimulant pills, including amphetamines, methylphenidate, and modafinil, are used to treat sleepiness but may increase irritability and will not improve cognitive abnormalities. Because of similarities between Kleine-Levin syndrome and certain mood disorders, lithium and carbamazepine may be prescribed and, in some cases, have been shown to prevent further episodes. This disorder should be differentiated from cyclic re-occurrence of sleepiness during the premenstrual period in teen-aged girls, which may be controlled with birth control pills. It also should be differentiated from encephalopathy, recurrent depression, or psychosis.
5. `NIDDK:0000107:0000107-8#0` — label: **not relevant**; judge says passage is not relevant; score 0.4287
   > The problems of gastroparesis can include - severe dehydration due to persistent vomiting - gastroesophageal reflux disease (GERD), which is GER that occurs more than twice a week for a few weeks; GERD can lead to esophagitis irritation of the esophagus - bezoars, which can cause nausea, vomiting, obstruction, or interfere with absorption of some medications in pill form - difficulty managing blood glucose levels in people with diabetes - malnutrition due to poor absorption of nutrients or a low calorie intake - decreased quality of life, including work absences due to severe symptoms

**Generated answer:**

```
## Summary
To prevent Cyclic Vomiting Syndrome (CVS), individuals should avoid known triggers, ensure adequate sleep, treat any sinus problems or allergies, manage stress and anxiety, and avoid specific foods that may provoke episodes. Consulting a healthcare provider for potential medication to prevent migraines may also be beneficial.

## Key Points
1. Avoiding known triggers, getting adequate sleep, treating sinus problems or allergies, and managing stress and anxiety can help prevent CVS [1].
2. Avoiding foods that trigger episodes or contain additives is recommended for prevention [1].
3. A healthcare provider may prescribe medications to prevent migraines in individuals with CVS [1].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / supported] To prevent Cyclic Vomiting Syndrome (CVS), individuals should avoid known triggers, ensure adequate sleep, treat any sinus problems or allergies, manage stress and anxiety, and avoid specific foods that may provoke episodes.
- [supported / supported] Consulting a healthcare provider for potential medication to prevent migraines may also be beneficial.
- [supported / supported] Avoiding known triggers, getting adequate sleep, treating sinus problems or allergies, and managing stress and anxiety can help prevent CVS.
- [supported / supported] Avoiding foods that trigger episodes or contain additives is recommended for prevention.
- [supported / supported] A healthcare provider may prescribe medications to prevent migraines in individuals with CVS.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | answered | 0.79 | 0.20 | 1.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 9: curated / hybrid

*Why this case is here:* random draw for this configuration.

**Query** (`NINDS:0000242:0000242-3`): What is the outlook for Progressive Supranuclear Palsy ?

**Reference answer (MedQuAD; never shown to the generator):** PSP gets progressively worse but is not itself directly life-threatening. It does, however, predispose patients to serious complications such as pneumonia secondary to difficulty in swallowing (dysphagia). The most common complications are choking and pneumonia, head injury, and fractures caused by falls. The most common cause of death is pneumonia. With good attention to medical and nutritional needs, however, most PSP patients live well into their 70s and beyond.

**Relevant chunks in this KB:** 1 strict, 1 lenient.

**Retrieved context, as shown to the generator:**

1. `GHR:0000832:0000832-2#0` — label: **not relevant**; judge says passage is relevant; score 0.0328
   > The exact prevalence of progressive supranuclear palsy is unknown. It may affect about 6 in 100,000 people worldwide.
2. `GARD:0001333:0001333-2#2` — label: **not relevant**; judge says passage is not relevant; score 0.0299
   > individuals with CPEO. The Human Phenotype Ontology provides the following list of signs and symptoms for Chronic progressive external ophthalmoplegia. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Adult onset - Autosomal dominant inheritance - Bradykinesia - Cataract - Decreased activity of cytochrome C oxidase in muscle tissue - Depression - Dysarthria - Dysphagia - EMG: myopathic abnormalities - Exercise intolerance - Facial palsy - Gait ataxia - Gastroparesis - Hypergonadotropic hypogonadism - Hyporeflexia - Impaired distal proprioception - Impaired distal vibration sensation
3. `GARD:0004742:0004742-1#1` — label: **not relevant**; judge says passage is not relevant; score 0.0290
   > - Hypokinesia - Hyposmia - Mask-like facies - Myoclonus - Paraparesis - Parkinsonism - Parkinsonism with favorable response to dopaminergic medication - Postural instability - Psychotic episodes - Rapidly progressive - Rigidity - Slow saccadic eye movements - Spasticity - Supranuclear gaze palsy - Torticollis - The Human Phenotype Ontology (HPO) has collected information on how often a sign or symptom occurs in a condition. Much of this information comes from Orphanet, a European rare disease database. The frequency of a sign or symptom is usually listed as a rough estimate of the percentage of patients who have that feature. The frequency may also be listed as a fraction. The first number of the fraction is how many people had
4. `SeniorHealth:0000052:0000052-6#2` — label: **not relevant**; judge says passage is not relevant; score 0.0275
   > gene activity in similar diseases such as progressive supranuclear palsy. In addition to identifying new genes for Parkinson's disease, researchers are trying to learn about the function of genes known to be associated with the disease, and about how gene mutations cause disease. Effects of Environmental Toxins Scientists continue to study environmental toxins such as pesticides and herbicides that can cause Parkinson's symptoms in animals. They have found that exposing rodents to the pesticide rotenone and several other agricultural chemicals can cause cellular and behavioral changes that mimic those seen in Parkinson's. Role of Lewy Bodies Other studies focus on how Lewy bodies form and what role they play in Parkinson's disease. Some studies suggest that Lewy bodies are a
5. `GARD:0001333:0001333-2#0` — label: **not relevant**; judge says passage is not relevant; score 0.0257
   > What are the signs and symptoms of Chronic progressive external ophthalmoplegia? The signs and symptoms of chronic progressive external ophthalmoplegia (CPEO) typically begin in young adults between the ages of 18 and 40. The most common symptoms in affected individuals include drooping eyelids (ptosis) and weakness or paralysis of the eye muscles (ophthalmoplegia). The condition may be unilateral (affecting one eye) or bilateral (affecting both eyes). Some affected individuals also have weakness of the skeletal muscles (myopathy), specifically of the arms, legs, and/or neck. This may be especially noticeable during exercise. Muscle weakness may also cause difficulty swallowing (dysphagia). Sometimes, CPEO may be associated with other signs and symptoms. In these cases, the condition is referred to as "progressive external

**Generated answer:**

```
## Summary
The outlook for Progressive Supranuclear Palsy (PSP) is generally poor, as it is a rapidly progressive neurodegenerative disorder. The exact prevalence is unknown, but it may affect about 6 in 100,000 people worldwide, indicating its rarity and the challenges in managing the condition effectively.

## Key Points
1. Progressive Supranuclear Palsy is a rapidly progressive neurodegenerative disorder. [3]
2. The exact prevalence of Progressive Supranuclear Palsy is unknown, but it may affect about 6 in 100,000 people worldwide. [1]

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [unsupported / unsupported] The outlook for Progressive Supranuclear Palsy (PSP) is generally poor, as it is a rapidly progressive neurodegenerative disorder.
- [supported / unsupported] The exact prevalence is unknown, but it may affect about 6 in 100,000 people worldwide, indicating its rarity and the challenges in managing the condition effectively.
- [unsupported / supported] Progressive Supranuclear Palsy is a rapidly progressive neurodegenerative disorder.
- [supported / unsupported] The exact prevalence of Progressive Supranuclear Palsy is unknown, but it may affect about 6 in 100,000 people worldwide.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | answered | 0.78 | 0.20 | 0.25 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 10: mixed / lexical

*Why this case is here:* random draw for this configuration.

**Query** (`MPlusHealthTopics:0000932:0000932-1`): What is (are) Urinary Tract Infections ?

**Reference answer (MedQuAD; never shown to the generator):** The urinary system is the body's drainage system for removing wastes and extra water. It includes two kidneys, two ureters, a bladder, and a urethra. Urinary tract infections (UTIs) are the second most common type of infection in the body. You may have a UTI if you notice - Pain or burning when you urinate - Fever, tiredness, or shakiness - An urge to urinate often - Pressure in your lower belly - Urine that smells bad or looks cloudy or reddish - Pain in your back or side below the ribs People of any age or sex can get UTIs. But about four times as many women get UTIs as men. You're also at higher risk if you have diabetes, need a tube to drain your bladder, or have a spinal cord injury. If you think you have a UTI it is important to see your doctor. Your doctor can tell if you have a UTI with a urine test. Treatment is with antibiotics. NIH: National Institute of Diabetes and Digestive and Kidney Diseases

**Relevant chunks in this KB:** 2 strict, 2 lenient.

**Retrieved context, as shown to the generator:**

1. `CDC:0000424:0000424-3#1` — label: **not relevant**; judge says passage is not relevant; score 21.6097
   > found in the environment. These bacteria can sometimes cause infections. Vancomycin is an antibiotic that is used to treat some drug-resistant infections caused by enterococci. In some instances, enterococci have become resistant to this drug and thus are called vancomycin-resistant enterococci (VRE). Most VRE infections occur in hospitals. Top of page What types of infections does VRE cause? VRE can live in the human intestines and female genital tract without causing disease (often called colonization). However, sometimes it can cause infections of the urinary tract, the bloodstream, or of wounds associated with catheters or surgical procedures. Top of page Are certain people at risk of getting VRE? The following persons are at increased risk becoming infected with VRE: People who
2. `CDC:0000424:0000424-5#1` — label: **not relevant**; judge says passage is not relevant; score 21.6097
   > found in the environment. These bacteria can sometimes cause infections. Vancomycin is an antibiotic that is used to treat some drug-resistant infections caused by enterococci. In some instances, enterococci have become resistant to this drug and thus are called vancomycin-resistant enterococci (VRE). Most VRE infections occur in hospitals. Top of page What types of infections does VRE cause? VRE can live in the human intestines and female genital tract without causing disease (often called colonization). However, sometimes it can cause infections of the urinary tract, the bloodstream, or of wounds associated with catheters or surgical procedures. Top of page Are certain people at risk of getting VRE? The following persons are at increased risk becoming infected with VRE: People who
3. `NIDDK:0000215:0000215-10#0` — label: **not relevant**; judge says passage is relevant; score 17.9718
   > Once the infection has cleared, more tests may be recommended to check for abnormalities in the urinary tract. Repeated infections in an abnormal urinary tract may cause kidney damage. The kinds of tests ordered will depend on the child and the type of urinary infection. Because no single test can tell everything about the urinary tract that might be important, more than one of the tests listed below may be needed. - Kidney and bladder ultrasound. Ultrasound uses a device, called a transducer, that bounces safe, painless sound waves off organs to create an image of their structure. The procedure is performed in a health care providers office, outpatient center, or hospital by a specially trained technician, and the images
4. `CDC:0000424:0000424-3#0` — label: **not relevant**; judge says passage is not relevant; score 17.6705
   > On this Page General Information What is vancomycin-resistant enterococci? What types of infections does vancomycin-resistant enterococci cause? Are certain people at risk of getting vancomycin-resistant enterococci? What is the treatment for vancomycin-resistant enterococci? How is vancomycin-resistant enterococci spread? How can patients prevent the spread of vancomycin-resistant enterococci? What should a patient do if they think they have vancomycin-resistant enterococci? Recommendations and Guidelines General Information For more images of this bacterium, search the Public Health Image Library What is vancomycin-resistant enterococci? Enteroccocci are bacteria that are normally present in the human intestines and in the female genital tract and are often found in the environment. These bacteria can sometimes cause infections. Vancomycin is an antibiotic that is used to treat some
5. `CDC:0000424:0000424-5#0` — label: **not relevant**; judge says passage is not relevant; score 17.6705
   > On this Page General Information What is vancomycin-resistant enterococci? What types of infections does vancomycin-resistant enterococci cause? Are certain people at risk of getting vancomycin-resistant enterococci? What is the treatment for vancomycin-resistant enterococci? How is vancomycin-resistant enterococci spread? How can patients prevent the spread of vancomycin-resistant enterococci? What should a patient do if they think they have vancomycin-resistant enterococci? Recommendations and Guidelines General Information For more images of this bacterium, search the Public Health Image Library What is vancomycin-resistant enterococci? Enteroccocci are bacteria that are normally present in the human intestines and in the female genital tract and are often found in the environment. These bacteria can sometimes cause infections. Vancomycin is an antibiotic that is used to treat some

**Generated answer:**

```
## Summary
Urinary Tract Infections (UTIs) are infections that can occur in the urinary tract, which includes the kidneys, bladder, and urethra. They can be caused by bacteria, including enterococci, which are normally present in the intestines and female genital tract but can lead to infections under certain conditions.

## Key Points
1. Urinary Tract Infections (UTIs) can occur in the urinary tract, which includes the kidneys, bladder, and urethra [3].
2. Enterococci, bacteria that are normally present in the human intestines and female genital tract, can sometimes cause urinary tract infections [1][4].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [unsupported / supported] Urinary Tract Infections (UTIs) are infections that can occur in the urinary tract, which includes the kidneys, bladder, and urethra.
- [supported / unsupported] They can be caused by bacteria, including enterococci, which are normally present in the intestines and female genital tract but can lead to infections under certain conditions.
- [unsupported / supported] Urinary Tract Infections (UTIs) can occur in the urinary tract, which includes the kidneys, bladder, and urethra.
- [supported / unsupported] Enterococci, bacteria that are normally present in the human intestines and female genital tract, can sometimes cause urinary tract infections.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | answered | 0.62 | 0.20 | 0.50 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 11: mixed / dense

*Why this case is here:* random draw for this configuration.

**Query** (`GHR:0000871:0000871-3`): What are the genetic changes related to retinitis pigmentosa ?

**Reference answer (MedQuAD; never shown to the generator):** Mutations in more than 60 genes are known to cause nonsyndromic retinitis pigmentosa. More than 20 of these genes are associated with the autosomal dominant form of the disorder. Mutations in the RHO gene are the most common cause of autosomal dominant retinitis pigmentosa, accounting for 20 to 30 percent of all cases. At least 35 genes have been associated with the autosomal recessive form of the disorder. The most common of these is USH2A; mutations in this gene are responsible for 10 to 15 percent of all cases of autosomal recessive retinitis pigmentosa. Changes in at least six genes are thought to cause the X-linked form of the disorder. Together, mutations in the RPGR and RP2 genes account for most cases of X-linked retinitis pigmentosa. The genes associated with retinitis pigmentosa play essential roles in the structure and function of specialized light receptor cells (photoreceptors) in the retina. The retina contains two types of photoreceptors, rods and cones. Rods are responsible for vision in low light, while cones provide vision in bright light, including color vision. Mutations in any of the genes responsible for retinitis pigmentosa lead to a gradual loss of rods and  …[truncated]

**Relevant chunks in this KB:** 3 strict, 3 lenient.

**Retrieved context, as shown to the generator:**

1. `GHR:0000871:0000871-3#0` — label: **GOLD**; judge says passage is relevant; score 0.6721
   > Mutations in more than 60 genes are known to cause nonsyndromic retinitis pigmentosa. More than 20 of these genes are associated with the autosomal dominant form of the disorder. Mutations in the RHO gene are the most common cause of autosomal dominant retinitis pigmentosa, accounting for 20 to 30 percent of all cases. At least 35 genes have been associated with the autosomal recessive form of the disorder. The most common of these is USH2A; mutations in this gene are responsible for 10 to 15 percent of all cases of autosomal recessive retinitis pigmentosa. Changes in at least six genes are thought to cause the X-linked form of the disorder. Together, mutations in the RPGR and RP2 genes account for
2. `GHR:0000871:0000871-3#1` — label: **GOLD**; judge says passage is relevant; score 0.6613
   > are thought to cause the X-linked form of the disorder. Together, mutations in the RPGR and RP2 genes account for most cases of X-linked retinitis pigmentosa. The genes associated with retinitis pigmentosa play essential roles in the structure and function of specialized light receptor cells (photoreceptors) in the retina. The retina contains two types of photoreceptors, rods and cones. Rods are responsible for vision in low light, while cones provide vision in bright light, including color vision. Mutations in any of the genes responsible for retinitis pigmentosa lead to a gradual loss of rods and cones in the retina. The progressive degeneration of these cells causes the characteristic pattern of vision loss that occurs in people with retinitis pigmentosa. Rods
3. `GHR:0000871:0000871-3#2` — label: **GOLD**; judge says passage is relevant; score 0.6194
   > progressive degeneration of these cells causes the characteristic pattern of vision loss that occurs in people with retinitis pigmentosa. Rods typically break down before cones, which is why night vision impairment is usually the first sign of the disorder. Daytime vision is disrupted later, as both rods and cones are lost. Some of the genes associated with retinitis pigmentosa are also associated with other eye diseases, including a condition called cone-rod dystrophy. Cone-rod dystrophy has signs and symptoms similar to those of retinitis pigmentosa. However, cone-rod dystrophy is characterized by deterioration of the cones first, followed by the rods, so daylight and color vision are affected before night vision.
4. `GHR:0001057:0001057-3#0` — label: **not relevant**; judge says passage is not relevant; score 0.5986
   > Mutations in the RS1 gene cause most cases of X-linked juvenile retinoschisis. The RS1 gene provides instructions for making a protein called retinoschisin, which is found in the retina. Studies suggest that retinoschisin plays a role in the development and maintenance of the retina. The protein is probably involved in the organization of cells in the retina by attaching cells together (cell adhesion). RS1 gene mutations result in a decrease in or complete loss of functional retinoschisin, which disrupts the maintenance and organization of cells in the retina. As a result, tiny splits (schisis) or tears form in the retina. This damage often forms a "spoke-wheel" pattern in the macula, which can be seen during an eye examination. In half
5. `GARD:0003332:0003332-1#1` — label: **not relevant**; judge says passage is not relevant; score 0.5551
   > can lead to blindness. Juvenile retinoschisis is caused by mutations in the RS1 gene. It is inherited in an X-linked recessive pattern. Low-vision aids can be helpful. Surgery may be needed for some complications.

**Generated answer:**

```
## Summary
Retinitis pigmentosa is caused by mutations in over 60 genes, with more than 20 associated with the autosomal dominant form, where the RHO gene is the most common. The autosomal recessive form is primarily linked to the USH2A gene, while the X-linked form is mainly due to mutations in the RPGR and RP2 genes. These genetic changes lead to the degeneration of photoreceptor cells in the retina, resulting in progressive vision loss.

## Key Points
1. Mutations in more than 60 genes are known to cause nonsyndromic retinitis pigmentosa, with specific genes linked to different inheritance patterns: RHO for autosomal dominant, USH2A for autosomal recessive, and RPGR/RP2 for X-linked forms [1][2].
2. The genetic mutations lead to the gradual loss of rods and cones in the retina, causing characteristic vision loss associated with retinitis pigmentosa [2][3].

## Disclaimer
This summary is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / supported] Retinitis pigmentosa is caused by mutations in over 60 genes, with more than 20 associated with the autosomal dominant form, where the RHO gene is the most common.
- [supported / supported] The autosomal recessive form is primarily linked to the USH2A gene, while the X-linked form is mainly due to mutations in the RPGR and RP2 genes.
- [supported / supported] These genetic changes lead to the degeneration of photoreceptor cells in the retina, resulting in progressive vision loss.
- [supported / supported] Mutations in more than 60 genes are known to cause nonsyndromic retinitis pigmentosa, with specific genes linked to different inheritance patterns: RHO for autosomal dominant, USH2A for autosomal recessive, and RPGR/RP2 for X-linked forms.
- [supported / supported] The genetic mutations lead to the gradual loss of rods and cones in the retina, causing characteristic vision loss associated with retinitis pigmentosa.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | answered | 0.74 | 0.60 | 1.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 12: mixed / hybrid

*Why this case is here:* random draw for this configuration.

**Query** (`GARD:0006312:0006312-4`): How to diagnose VACTERL association ?

**Reference answer (MedQuAD; never shown to the generator):** Is genetic testing available for VACTERL association? Because there is no known cause of VACTERL association, clinical genetic testing is not available for the condition. If an individual has a specific diagnosis of another syndrome or genetic condition in addition to the features of VACTERL association, genetic testing may be available for that condition. GeneTests lists the names of laboratories that are performing genetic testing for VACTERL association. Although no clinical laboratories are listed for this condition, there are some research laboratories performing genetic testing; to see a list of these laboratories, click here. Please note: Most of the laboratories listed through GeneTests do not accept direct contact from patients and their families; therefore, if you are interested in learning more, you will need to work with a health care provider or a genetics professional. How is VACTERL association diagnosed? Prenatal diagnosis of VACTERL association can be challenging because certain component features of the condition can be difficult to detect prior to birth. Therefore, the diagnosis of VACTERL association is typically based on features that are seen when a baby is bo …[truncated]

**Relevant chunks in this KB:** 4 strict, 4 lenient.

**Retrieved context, as shown to the generator:**

1. `GARD:0006312:0006312-4#1` — label: **GOLD**; judge says passage is relevant; score 0.0328
   > listed through GeneTests do not accept direct contact from patients and their families; therefore, if you are interested in learning more, you will need to work with a health care provider or a genetics professional. How is VACTERL association diagnosed? Prenatal diagnosis of VACTERL association can be challenging because certain component features of the condition can be difficult to detect prior to birth. Therefore, the diagnosis of VACTERL association is typically based on features that are seen when a baby is born or in the first few days of life. The diagnosis is based on having at least three of the following features (which make up the acronym VACTERL): vertebral defects, commonly accompanied by rib anomalies; imperforate anus or anal
2. `GARD:0006312:0006312-4#0` — label: **GOLD**; judge says passage is relevant; score 0.0320
   > Is genetic testing available for VACTERL association? Because there is no known cause of VACTERL association, clinical genetic testing is not available for the condition. If an individual has a specific diagnosis of another syndrome or genetic condition in addition to the features of VACTERL association, genetic testing may be available for that condition. GeneTests lists the names of laboratories that are performing genetic testing for VACTERL association. Although no clinical laboratories are listed for this condition, there are some research laboratories performing genetic testing; to see a list of these laboratories, click here. Please note: Most of the laboratories listed through GeneTests do not accept direct contact from patients and their families; therefore, if you are interested in learning
3. `GHR:0000327:0000327-3#1` — label: **not relevant**; judge says passage is relevant; score 0.0315
   > born with EA/TEF. EA/TEF also occurs in VACTERL association, a multifactorial condition. VACTERL is an acronym that stands for vertebral defects, anal atresia, cardiac defects, tracheoesophageal fistula, renal anomalies, and limb abnormalities. People diagnosed with VACTERL association typically have at least three of these features; between 50 and 80 percent have a tracheoesophageal fistula.
4. `GARD:0006312:0006312-4#2` — label: **GOLD**; judge says passage is relevant; score 0.0311
   > the following features (which make up the acronym VACTERL): vertebral defects, commonly accompanied by rib anomalies; imperforate anus or anal atresia; cardiac (heart) defects; tracheo-esophageal fistula with or without esophageal atresia; renal (kidney) anomalies including renal agenesis, horseshoe kidney, and cystic and/or dysplastic kidneys; and limb abnormalities. Additional types of abnormalities have also been reported in affected individuals and may be used as clues in considering a diagnosis of other conditions with overlapping features. Depending on the features present, some other conditions that may be considered when diagnosing a child who has features of VACTERL association (differential diagnosis) may include Baller-Gerold syndrome, CHARGE syndrome, Currarino disease, 22q11.2 microdeletion syndrome, Fanconi anemia, Feingold syndrome, Fryns syndrome, MURCS association, oculo-auriculo-vertebral spectrum, Opitz
5. `GARD:0006312:0006312-4#3` — label: **GOLD**; judge says passage is not relevant; score 0.0308
   > Baller-Gerold syndrome, CHARGE syndrome, Currarino disease, 22q11.2 microdeletion syndrome, Fanconi anemia, Feingold syndrome, Fryns syndrome, MURCS association, oculo-auriculo-vertebral spectrum, Opitz G/BBB syndrome, Pallister-Hall syndrome, Townes-Brocks syndrome, and VACTERL with hydrocephalus.

**Generated answer:**

```
## Summary
VACTERL association is diagnosed based on the presence of at least three specific features observed at birth or shortly thereafter. These features include vertebral defects, anal atresia, cardiac defects, tracheoesophageal fistula, renal anomalies, and limb abnormalities.

## Key Points
1. The diagnosis of VACTERL association is typically based on having at least three of the following features: vertebral defects, anal atresia, cardiac defects, tracheoesophageal fistula, renal anomalies, and limb abnormalities [1][3][4].
2. Prenatal diagnosis of VACTERL association can be challenging, as many features may not be detectable before birth [1].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / supported] VACTERL association is diagnosed based on the presence of at least three specific features observed at birth or shortly thereafter.
- [supported / supported] These features include vertebral defects, anal atresia, cardiac defects, tracheoesophageal fistula, renal anomalies, and limb abnormalities.
- [supported / supported] The diagnosis of VACTERL association is typically based on having at least three of the following features: vertebral defects, anal atresia, cardiac defects, tracheoesophageal fistula, renal anomalies, and limb abnormalities.
- [supported / supported] Prenatal diagnosis of VACTERL association can be challenging, as many features may not be detectable before birth.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 1.00 | 1.00 | 0.96 | 1.00 | answered | 0.84 | 0.80 | 1.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 13: reduced / lexical

*Why this case is here:* random draw for this configuration.

**Query** (`NHLBI:0000067:0000067-1`): What is (are) Hemochromatosis ?

**Reference answer (MedQuAD; never shown to the generator):** Hemochromatosis (HE-mo-kro-ma-TO-sis) is a disease in which too much iron builds up in your body (iron overload). Iron is a mineral found in many foods. Too much iron is toxic to your body. It can poison your organs and cause organ failure. In hemochromatosis, iron can build up in most of your body's organs, but especially in the liver, heart, and pancreas. Too much iron in the liver can cause an enlarged liver, liver failure, liver cancer, or cirrhosis (sir-RO-sis). Cirrhosis is scarring of the liver, which causes the organ to not work well. Too much iron in the heart can cause irregular heartbeats called arrhythmias (ah-RITH-me-ahs) and heart failure. Too much iron in the pancreas can lead to diabetes. If hemochromatosis isn't treated, it may even cause death. Overview The two types of hemochromatosis are primary and secondary. Primary hemochromatosis is caused by a defect in the genes that control how much iron you absorb from food. Secondary hemochromatosis usually is the result of another disease or condition that causes iron overload. Most people who have primary hemochromatosis inherit it from their parents. If you inherit two hemochromatosis genesone from each parentyou're  …[truncated]

**Relevant chunks in this KB:** 5 strict, 6 lenient.

**Retrieved context, as shown to the generator:**

1. `NIDDK:0000185:0000185-1#0` — label: **lenient-relevant**; judge says passage is relevant; score 15.0814
   > Hemochromatosis is the most common form of iron overload disease. Too much iron in the body causes hemochromatosis. Iron is important because it is part of hemoglobin, a molecule in the blood that transports oxygen from the lungs to all body tissues. However, too much iron in the body leads to iron overloada buildup of extra iron that, without treatment, can damage organs such as the liver, heart, and pancreas; endocrine glands; and joints. The three types of hemochromatosis are primary hemochromatosis, also called hereditary hemochromatosis; secondary hemochromatosis; and neonatal hemochromatosis.
2. `NHLBI:0000067:0000067-1#1` — label: **GOLD**; judge says passage is relevant; score 13.4207
   > can cause irregular heartbeats called arrhythmias (ah-RITH-me-ahs) and heart failure. Too much iron in the pancreas can lead to diabetes. If hemochromatosis isn't treated, it may even cause death. Overview The two types of hemochromatosis are primary and secondary. Primary hemochromatosis is caused by a defect in the genes that control how much iron you absorb from food. Secondary hemochromatosis usually is the result of another disease or condition that causes iron overload. Most people who have primary hemochromatosis inherit it from their parents. If you inherit two hemochromatosis genesone from each parentyou're at risk for iron overload and signs and symptoms of the disease. The two faulty genes cause your body to absorb more iron than usual from the
3. `GARD:0002767:0002767-2#0` — label: **not relevant**; judge says passage is not relevant; score 13.2743
   > What are the signs and symptoms of Hemochromatosis type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemochromatosis type 1. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Abnormal glucose tolerance - Alopecia - Amenorrhea - Arrhythmia - Arthropathy - Ascites - Autosomal recessive inheritance - Azoospermia - Cardiomegaly - Cardiomyopathy - Cirrhosis - Congestive heart failure - Diabetes mellitus - Elevated hepatic transaminases - Hepatocellular carcinoma - Hepatomegaly - Hyperpigmentation of the skin - Hypogonadotrophic hypogonadism
4. `NHLBI:0000067:0000067-4#0` — label: **not relevant**; judge says passage is relevant; score 13.0609
   > Hemochromatosis is one of the most common genetic diseases in the United States. It's most common in Caucasians of Northern European descent. The disease is less common in African Americans, Hispanics, Asians, and American Indians. Primary hemochromatosis is more common in men than in women. Also, older people are more likely to develop the disease than younger people. In fact, signs and symptoms usually don't occur in men until they're 40 to 60 years old. In women, signs and symptoms usually don't occur until after the age of 50 (after menopause). Young children rarely develop hemochromatosis. Inheriting two faulty HFE genes (one from each parent) is the major risk factor for hemochromatosis. However, many people who have two copies of
5. `GARD:0002766:0002766-4#2` — label: **not relevant**; judge says passage is relevant; score 12.6139
   > 4 hemochromatosis has one affected parent. Acquired hemochromatosis is not inherited. Neonatal hemochromatosis is also not inherited, but it does appear to be familial. A woman with an affected child has approximately an 80% chance to have another affected child. However, this likelihood of recurrence is not explained by normal inheritance patterns. The underlying cause of this type is not fully understood.

**Generated answer:**

```
## Summary
Hemochromatosis is the most common form of iron overload disease, characterized by excessive iron accumulation in the body, which can damage organs such as the liver, heart, and pancreas. There are three types: primary (hereditary), secondary (resulting from another condition), and neonatal hemochromatosis. Primary hemochromatosis is often inherited and caused by genetic defects affecting iron absorption.

## Key Points
1. Hemochromatosis is caused by too much iron in the body, leading to potential organ damage if untreated [1]. 
2. There are three types of hemochromatosis: primary, secondary, and neonatal [1]. 
3. Primary hemochromatosis is typically inherited and results from a genetic defect affecting iron absorption [2]. 

## Disclaimer
This summary is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / supported] Hemochromatosis is the most common form of iron overload disease, characterized by excessive iron accumulation in the body, which can damage organs such as the liver, heart, and pancreas.
- [supported / unsupported] There are three types: primary (hereditary), secondary (resulting from another condition), and neonatal hemochromatosis.
- [supported / supported] Primary hemochromatosis is often inherited and caused by genetic defects affecting iron absorption.
- [supported / supported] Hemochromatosis is caused by too much iron in the body, leading to potential organ damage if untreated.
- [supported / unsupported] There are three types of hemochromatosis: primary, secondary, and neonatal.
- [supported / supported] Primary hemochromatosis is typically inherited and results from a genetic defect affecting iron absorption.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.20 | 0.50 | 0.21 | 1.00 | answered | 0.70 | 0.80 | 0.67 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 14: reduced / dense

*Why this case is here:* random draw for this configuration.

**Query** (`NINDS:0000073:0000073-2`): What are the treatments for Cerebro-Oculo-Facio-Skeletal Syndrome (COFS) ?

**Reference answer (MedQuAD; never shown to the generator):** Treatment is supportive and symptomatic. Individuals with the disorder often require tube feeding. Because COFS is genetic, genetic counseling is available.

**Relevant chunks in this KB:** 0 strict, 0 lenient.

**Retrieved context, as shown to the generator:**

1. `NINDS:0000081:0000081-2#0` — label: **not relevant**; judge says passage is not relevant; score 0.5019
   > There is no definitive treatment for colpocephaly. Anticonvulsant medications are often prescribed to prevent seizures, and doctors rely on exercise therapies and orthopedic appliances to reduce shrinkage or shortening of muscles.
2. `GARD:0000838:0000838-2#0` — label: **not relevant**; judge says passage is not relevant; score 0.4921
   > What are the signs and symptoms of Branchiooculofacial syndrome? The characteristic signs and symptoms of BOFS include skin defects, eye abnormalities, and distinctive facial features. These features vary among affected individuals. The skin defects include proliferation of blood vessels (hemangiomatous) in the lower neck or upper chest; lumps in the area of the neck or collarbone (branchial cleft sinuses); and linear skin lesions behind the ears. Eye abnormalities can include microphthalmia, coloboma, and strabismus. The distinctive facial features can include widely spaced eyes; the presence of a pseudocleft of the upper lip resembling a poorly repaired cleft lip; a malformed nose with a broad bridge and flattened tip; blockage of the tear ducts (lacrimal duct obstruction); and malformed ears. Often,
3. `GARD:0004627:0004627-1#1` — label: **not relevant**; judge says passage is not relevant; score 0.4636
   > muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person.
4. `GARD:0000138:0000138-1#1` — label: **not relevant**; judge says passage is not relevant; score 0.4423
   > No treatment has been proven to improve the visual outcome of AZOOR; however, systemic corticosteroids are the most commonly used therapy.
5. `GARD:0004611:0004611-1#1` — label: **not relevant**; judge says passage is not relevant; score 0.4402
   > Depending on the genetic cause, OI may be inherited in an autosomal dominant (more commonly) or autosomal recessive manner. Treatment is supportive and aims to decrease the number of fractures and disabilities.

**Generated answer:**

```
## Summary
The provided context does not specify any treatments for Cerebro-Oculo-Facio-Skeletal Syndrome (COFS). 

## Key Points
1. There is no definitive treatment for colpocephaly, which is mentioned in relation to other conditions but not COFS specifically [1].
2. Treatments for other syndromes mentioned, such as Branchiooculofacial syndrome and Osteopathia striata cranial sclerosis, are based on symptoms and genetic causes, but these do not apply directly to COFS [2][3][5].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [no_information / no_information] The provided context does not specify any treatments for Cerebro-Oculo-Facio-Skeletal Syndrome (COFS).
- [unsupported / unsupported] There is no definitive treatment for colpocephaly, which is mentioned in relation to other conditions but not COFS specifically.
- [unsupported / unsupported] Treatments for other syndromes mentioned, such as Branchiooculofacial syndrome and Osteopathia striata cranial sclerosis, are based on symptoms and genetic causes, but these do not apply directly to COFS.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.00 | undefined | undefined | undefined | 0.00 | partial | 0.86 | 0.00 | 0.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:

---

## Case 15: reduced / hybrid

*Why this case is here:* random draw for this configuration.

**Query** (`NHLBI:0000006:0000006-3`): Who is at risk for Aplastic Anemia? ?

**Reference answer (MedQuAD; never shown to the generator):** Aplastic anemia is a rare but serious blood disorder. People of all ages can develop aplastic anemia. However, it's most common in adolescents, young adults, and the elderly. Men and women are equally likely to have it. The disorder is two to three times more common in Asian countries. Your risk of aplastic anemia is higher if you: Have been exposed to toxins Have taken certain medicines or had radiation or chemotherapy (treatments for cancer) Have certain infectious diseases, autoimmune disorders, or inherited conditions For more information, go to "What Causes Aplastic Anemia?"

**Relevant chunks in this KB:** 1 strict, 1 lenient.

**Retrieved context, as shown to the generator:**

1. `NHLBI:0000006:0000006-3#0` — label: **GOLD**; judge says passage is relevant; score 0.0328
   > Aplastic anemia is a rare but serious blood disorder. People of all ages can develop aplastic anemia. However, it's most common in adolescents, young adults, and the elderly. Men and women are equally likely to have it. The disorder is two to three times more common in Asian countries. Your risk of aplastic anemia is higher if you: Have been exposed to toxins Have taken certain medicines or had radiation or chemotherapy (treatments for cancer) Have certain infectious diseases, autoimmune disorders, or inherited conditions For more information, go to "What Causes Aplastic Anemia?"
2. `NHLBI:0000006:0000006-5#0` — label: **not relevant**; judge says passage is not relevant; score 0.0318
   > Your doctor will diagnose aplastic anemia based on your medical and family histories, a physical exam, and test results. Once your doctor knows the cause and severity of the condition, he or she can create a treatment plan for you. Specialists Involved If your primary care doctor thinks you have aplastic anemia, he or she may refer you to a hematologist. A hematologist is a doctor who specializes in treating blood diseases and disorders. Medical and Family Histories Your doctor may ask questions about your medical history, such as whether: You've had anemia or a condition that can cause anemia You have shortness of breath, dizziness, headaches, or other signs and symptoms of anemia You've been exposed to certain toxins
3. `NHLBI:0000006:0000006-5#2` — label: **not relevant**; judge says passage is not relevant; score 0.0313
   > checking for pale or yellowish skin and signs of bleeding or infection. Your doctor may listen to your heart and lungs for abnormal heartbeats and breathing sounds. He or she also may feel your abdomen to check the size of your liver and feel your legs for swelling. Diagnostic Tests Many tests are used to diagnose aplastic anemia. These tests help: Confirm a diagnosis of aplastic anemia, look for its cause, and find out how severe it is Rule out other conditions that may cause similar symptoms Check for paroxysmal nocturnal hemoglobinuria (PNH) Complete Blood Count Often, the first test used to diagnose aplastic anemia is a complete blood count (CBC). The CBC measures many parts of your blood. This
4. `NHLBI:0000006:0000006-5#6` — label: **not relevant**; judge says passage is not relevant; score 0.0310
   > doctor removes a small amount of bone marrow tissue through a needle. The tissue is checked for the number and types of cells in the bone marrow. In aplastic anemia, the bone marrow has a lower than normal number of all three types of blood cells. Other Tests Other conditions can cause symptoms similar to those of aplastic anemia. Thus, other tests may be needed to rule out those conditions. These tests may include: X ray, computed tomography (CT) scan, or an ultrasound imaging test. These tests can show enlarged lymph nodes in your abdomen. Enlarged lymph nodes may be a sign of blood cancer. Doctors also may use these tests to look at the kidneys and the bones in
5. `NHLBI:0000006:0000006-5#8` — label: **not relevant**; judge says passage is not relevant; score 0.0305
   > in the blood. These tests can help rule out anemia caused by vitamin deficiency. Your doctor also may recommend blood tests for PNH and to check your immune system for proteins called antibodies. (Antibodies in the immune system that attack your bone marrow cells may cause aplastic anemia.)

**Generated answer:**

```
## Summary
Individuals at risk for aplastic anemia include people of all ages, particularly adolescents, young adults, and the elderly. Risk factors include exposure to toxins, certain medications, radiation or chemotherapy, and having specific infectious diseases, autoimmune disorders, or inherited conditions.

## Key Points
1. Aplastic anemia can develop in people of all ages, but it is most common in adolescents, young adults, and the elderly. Men and women are equally affected [1].
2. Higher risk factors for aplastic anemia include exposure to toxins, certain medications, radiation or chemotherapy, and specific infectious diseases, autoimmune disorders, or inherited conditions [1].

## Disclaimer
This information is not a substitute for professional medical advice.
```

**Judge's label for each claim** (context / reference answer):

- [supported / supported] Individuals at risk for aplastic anemia include people of all ages, particularly adolescents, young adults, and the elderly.
- [supported / supported] Risk factors include exposure to toxins, certain medications, radiation or chemotherapy, and having specific infectious diseases, autoimmune disorders, or inherited conditions.
- [supported / supported] Aplastic anemia can develop in people of all ages, but it is most common in adolescents, young adults, and the elderly. Men and women are equally affected.
- [supported / supported] Higher risk factors for aplastic anemia include exposure to toxins, certain medications, radiation or chemotherapy, and specific infectious diseases, autoimmune disorders, or inherited conditions.

| P@k | R@k | MRR | NDCG | faithfulness | abstention | answer relevance | context relevance | reference agreement |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | answered | 0.77 | 0.20 | 1.00 |

**Your judgement:** does the answer answer the question? ☐ yes ☐ partly ☐ no — is every claim in it supported by the retrieved context? ☐ yes ☐ no — do the scores above match what you see? ☐ yes ☐ no. Notes:
