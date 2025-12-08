# NOTE: Uncomment and implement the test runners you want
# Generated with https://github.com/flatland-association/flatland-benchmarks/blob/main/definitions/ai4realnet/gen_ai4realnet_benchmarks_sql.py
# from https://inesctecpt.sharepoint.com/:x:/r/sites/AI4REALNET/Shared%20Documents/General/WP4%20-%20Validation%20and%20impact%20assessment/Validation%20campaigns/Overview%20tests%20for%20KPI%20on%20validation%20campaign%20hub.xlsx?d=w947339379458465eaaf243a750315375&csf=1&web=1&e=RnrCdf
from ai4realnet_orchestrators.orchestrator import Orchestrator
from ai4realnet_orchestrators.railway.test_runner_kpi_af_029_railway import TestRunner_KPI_AF_029_Railway
from ai4realnet_orchestrators.railway.test_runner_kpi_nf_045_railway import TestRunner_KPI_NF_045_Railway
from ai4realnet_orchestrators.railway.test_runner_kpi_pf_026_railway import TestRunner_KPI_PF_026_Railway
from ai4realnet_orchestrators.railway.test_runner_kpi_rs_058_railway import TestRunner_KPI_RS_058_Railway

railway_orchestrator = Orchestrator(
  test_runners={

    #     # KPI-AS-001: Ability to anticipate (Railway)
    #     "8edbf9a1-09b4-478a-ac53-8fc903e70cc1": TestRunner_KPI_AS_001_Railway(
    #         test_id="8edbf9a1-09b4-478a-ac53-8fc903e70cc1", scenario_ids=['c7159f8d-781b-40fc-9efa-cd0e3a8b8d21'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-AS-009: Assistant disturbance (Railway)
    #     "d4bcbd11-740a-478f-8dbc-6d9b53fa6059": TestRunner_KPI_AS_009_Railway(
    #         test_id="d4bcbd11-740a-478f-8dbc-6d9b53fa6059", scenario_ids=['59d015ca-ca8a-4015-a113-35c182b301e4'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-DS-015: Decision support satisfaction (Railway)
    #     "22c590e5-01ad-4e96-9040-29a13ac9118f": TestRunner_KPI_DS_015_Railway(
    #         test_id="22c590e5-01ad-4e96-9040-29a13ac9118f", scenario_ids=['cbfe8dd3-e8df-464f-92cd-0adeab4a18b8'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-HS-022: Human motivation (Railway)
    #     "05e91ee6-744b-47ea-85a0-005e26b578e0": TestRunner_KPI_HS_022_Railway(
    #         test_id="05e91ee6-744b-47ea-85a0-005e26b578e0", scenario_ids=['ab770463-5832-4edd-9ab2-178b7ee46b74'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-HS-023: Human response time (Railway)
    #     "40398221-b288-46d9-895c-45d278f1b6a3": TestRunner_KPI_HS_023_Railway(
    #         test_id="40398221-b288-46d9-895c-45d278f1b6a3", scenario_ids=['35cf47d4-83f4-4eb7-ab9b-40c6fe7679a8'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-SS-031: Situation awareness (Railway)
    #     "307bce4c-635f-44c3-9ed2-6d4d45d1bbb2": TestRunner_KPI_SS_031_Railway(
    #         test_id="307bce4c-635f-44c3-9ed2-6d4d45d1bbb2", scenario_ids=['3131063f-5d0e-42bd-955d-5e875ceaac94'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-WS-040: Workload (Railway)
    #     "343bc5cf-0bab-4fdf-9760-266c5a738b08": TestRunner_KPI_WS_040_Railway(
    #         test_id="343bc5cf-0bab-4fdf-9760-266c5a738b08", scenario_ids=['c3ab2e6b-e8ed-420e-b6d2-4fd5dd407288'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-CS-049: Cognitive Performance & Stress (Railway)
    #     "452bb0df-9a9d-4475-ac7b-8e62659c0b13": TestRunner_KPI_CS_049_Railway(
    #         test_id="452bb0df-9a9d-4475-ac7b-8e62659c0b13", scenario_ids=['76d73b49-ff52-4b9a-96e2-cc664582c8e4'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
    #     ),
    #
    #     # KPI-AS-002: Acceptance (Railway)
    #     "ba7fabe3-6f42-4cc5-875e-b8fb45aa17f3": TestRunner_KPI_AS_002_Railway(
    #         test_id="ba7fabe3-6f42-4cc5-875e-b8fb45aa17f3", scenario_ids=['0d85a097-9170-42aa-99ea-d56b833c27cf'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
    #     ),
    #
    #     # KPI-AS-005: Agreement score (Railway)
    #     "fd01a013-c5aa-4ab2-9c01-ec608013c929": TestRunner_KPI_AS_005_Railway(
    #         test_id="fd01a013-c5aa-4ab2-9c01-ec608013c929", scenario_ids=['9db20a76-763a-4597-9586-cb217981e191'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
    #     ),
    #
    #     # KPI-CS-013: Comprehensibility (Railway)
    #     "3c249970-1e83-4faa-bcf4-9fdb63ed904d": TestRunner_KPI_CS_013_Railway(
    #         test_id="3c249970-1e83-4faa-bcf4-9fdb63ed904d", scenario_ids=['0ee42e0f-a284-4979-86e8-4e50c9bfcef7'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
    #     ),
    #
    #     # KPI-TS-038: Trust in AI solutions score (Railway)
    #     "57134a8f-d2bb-4a49-975c-4f6e1e07eb09": TestRunner_KPI_TS_038_Railway(
    #         test_id="57134a8f-d2bb-4a49-975c-4f6e1e07eb09", scenario_ids=['7b07ae08-9153-42cd-b1e9-6c03f3c1df31'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
    #     ),
    #
    #     # KPI-TS-039: Trust towards the AI tool (Railway)
    #     "51780641-4f2a-4095-b230-fdddc4bf31af": TestRunner_KPI_TS_039_Railway(
    #         test_id="51780641-4f2a-4095-b230-fdddc4bf31af", scenario_ids=['0e9f4bb5-6e39-4ccb-b5ae-81b9a1f91607'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
    #     ),
    #
    #     # KPI-HS-003: Human intervention frequency (Railway)
    #     "22b1dc11-7d3f-4219-adc9-b1eba58562a7": TestRunner_KPI_HS_003_Railway(
    #         test_id="22b1dc11-7d3f-4219-adc9-b1eba58562a7", scenario_ids=['6c383eec-31cd-4f3c-9296-cc5bb0d7f4c9'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
    #     ),
    #
    #     # KPI-SS-030: Significance of human revisions (Railway)
    #     "24eecb84-e881-459c-8116-a224b0253b70": TestRunner_KPI_SS_030_Railway(
    #         test_id="24eecb84-e881-459c-8116-a224b0253b70", scenario_ids=['eee4453e-477f-4057-ad7b-c1c2233ed108'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
    #     ),
    #
    #     # KPI-PS-089: Perceived decision novelty (Railway)
    #     "74c88d9b-f61b-4162-aefc-75a9d538b8a6": TestRunner_KPI_PS_089_Railway(
    #         test_id="74c88d9b-f61b-4162-aefc-75a9d538b8a6", scenario_ids=['b228ee9c-de35-4efd-b59a-7c4bb95ca127'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
    #     ),
    #
    #     # KPI-AS-006: AI co-learning capability (Railway)
    #     "a38618ca-6e9f-437d-a177-270f721ad1c6": TestRunner_KPI_AS_006_Railway(
    #         test_id="a38618ca-6e9f-437d-a177-270f721ad1c6", scenario_ids=['9fdd3565-45d6-4d83-9cda-478c56e94f26'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
    #     ),
    #
    #     # KPI-HS-021: Human learning (Railway)
    #     "c258c64f-1905-4d7f-93f0-d696c133978e": TestRunner_KPI_HS_021_Railway(
    #         test_id="c258c64f-1905-4d7f-93f0-d696c133978e", scenario_ids=['b4c9184c-c0d1-4a0c-bab1-9210bf8cb548'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
    #     ),
    #
    #     # KPI-DF-016: Delay reduction efficiency (Railway)
    #     "6ff3c588-357c-41a6-a45a-2bd946b158c8": TestRunner_KPI_DF_016_Railway(
    #         test_id="6ff3c588-357c-41a6-a45a-2bd946b158c8", scenario_ids=['ba7f9aac-5e96-4436-bae1-23629c4d153b'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
    #     ),
    #

    # KPI-PF-026: Punctuality (Railway)
    "98ceb866-5479-47e6-a735-81292de8ca65": TestRunner_KPI_PF_026_Railway(
      test_id="98ceb866-5479-47e6-a735-81292de8ca65", scenario_ids=[
        '5a60713d-01f2-4d32-9867-21904629e254', '0db72a40-43e8-477b-89b3-a7bd1224660a', '7def3118-2e9c-4de7-8d61-f0e76fbeee5d',
        '3ae60635-6995-4fb1-8309-61fded3d6fd8', 'eeef8445-723d-4740-b89f-4dbaf75f9ae6', '94af1ed1-3686-4a9e-99f5-3a7ad908f125',
        '8250d0e2-700e-4051-85c3-a8d0d95a5f0f', 'c58759a7-a64a-4cbf-970b-948bae0c2254', 'f94f517f-c0a4-4415-b726-186cdc75f9c6',
        'c0e2c3e0-c171-48dd-a312-5de070e3f937', '26fb51b9-2466-48ab-8c0a-89d9536d4c34', '66c23554-47bb-4268-b08e-518f6f163e9d',
        '08641d25-9b18-41bb-9cbc-4039b4ad24f0', '4d3dd85a-2b16-45de-b524-f83b4a58a2f4', 'bf9209c7-125c-42f9-b78e-4e5b7aacefcc',
        'f511044d-2378-4c7f-af92-45c78146bdef', 'f89cee49-e1ae-427d-ae42-5cc411661a1c', '262e43bd-bf35-4171-b38b-c77969db0b16',
        '61bd29ec-0b09-4067-ada0-b43e48a8ac9a', '92d22472-4696-40ba-924f-861a2f4343b6', '6fc5f67a-40fa-45ce-819e-35a85e08e560',
        '66bce513-502c-43b4-a155-8a16c410a7c6', 'eff645bf-7ea8-490d-ae8a-ebb0d16a774c', '8397e6d6-babc-469b-a239-7eabcbd510da',
        'c359f13c-d222-4b04-ad0a-2bb30fb9da5f', '97203764-6717-4ca6-bae9-c35c4eb38206', 'adc4bf52-096c-4369-a85f-c9bf4b86bc64',
        '72f93d48-ecef-4bf7-9d97-cb008b47e566', 'b470667b-d9c9-4af4-b64e-c32102c34387', '4aa9e1b8-8669-466e-b4b9-c7db2a098bec',
        '44354e67-e0b7-4faa-9385-2d6247c7a50c', '4bca2964-ef35-4b03-a47e-829bc9078374', '6190b6ac-a06b-4b07-8b82-0dfb1088663f',
        'a51d4e4a-c841-416f-9292-fc64dead758b', 'd08f6539-5c3e-4a98-93ae-3e344611e3a8', '8a2ec760-d2c2-4329-b373-3acd95395076',
        '9b136147-f560-45d4-abd8-9f7de7cd7570', '9723c2c5-ee55-441d-9dba-1a1dc23fdc5e', '3406bbe9-cb19-44b7-af26-460b0a117a6c',
        'ae0a88ad-3bfd-4201-b28a-e2c75d081cd5', '8b308495-7ea6-4ddc-acb4-56eb5b3aec12', 'a8f69dc4-04a1-434a-ad97-27c745561b6a',
        '8b244f56-50e1-411a-a7d8-a2b89dfab26e', '8e6419c1-6470-4272-9c4b-43d9fe19dd3d', 'ec503b6e-3682-4dcd-9dc7-b194b67283d9',
        '74fd9eab-d2e5-4222-8656-81fc2dde7c21', 'c16e54c1-33b2-45b8-95b0-33cc4f5400d5', 'c80effec-27b8-4103-b726-344a85f35407',
        '9bec9335-3dd5-4d88-b2ac-c5d711bcab36', '4a067d3c-75e6-4e91-a42d-cdf291016674', '1f6c674b-fdb5-40c8-bbe3-d924c2b7146e',
        '8bd15e2f-6089-4022-8383-26a36093dc80', '5b026a8a-e0dc-47d2-b2ce-d0bfc083e6c5', 'ab2e2bba-baa0-45de-b639-c7ebf29bf947',
        '8f8b6f67-d3c3-41ec-b60d-2924577e68f4', '3e59642a-7b20-4989-b9e6-01a35a82b9da', '0e4618c7-7ab7-4323-855c-7f95cbaef2d0',
        '4a59929b-b391-464f-bfa1-628f7a45ac36', '27f8ac34-3d09-4b68-9fb4-c51cfbfd09df', 'bacb4c11-11b4-4c76-910d-f92abc5b7a39',
        'a5dff3c8-902e-4cb9-8466-d277d0ed4d67', 'aca25feb-6254-40b3-8d40-3c805797c69b', 'deb21442-0f94-4ff3-b78d-8d418415d646',
        '6cf2cc89-d30e-4063-bced-051f3cdae92f', '84bcbff5-346f-452c-87ab-08ceff6364f2', '9acbe68e-2a45-420b-a142-34996dbcfb83',
        '42786e4c-c80e-40f5-8237-bafc5f39979d', '242a6240-b62c-48b4-a264-b6737e893fa5', '3c38a1d3-2340-43ed-ac0b-4b76c6588b92',
        'b89daede-405b-411a-a02b-ee32d7c9d020', '736301d9-0cff-4e25-af78-4d6f78b48cd5', '3b74e3a8-e740-4bb2-802f-2b2bedabad65',
        '61ee09b0-96fb-4562-bf7d-a01606de424c', '8d75b03c-2a34-4b2a-8408-d8db01a7ae1a', '81c01671-1a30-41cd-9f6c-25b2f9253da9',
        '01723b62-3f4d-4921-8904-752f092588f5', '5d44cb89-3c63-4716-8551-bd25de881f89', 'cf7c97fc-1c61-4c0d-ad1e-952bf6d6f23a',
        '60ddab54-dc12-4fbd-bbec-53b23d896c9d', '7eab41ad-8fd5-4a04-8c06-4d6c2016a594', '9e0aac9e-ddf9-4575-bf1c-d08a923e15fa',
        'befa97fb-2a74-4f2e-91c8-ea2879d08dcf', '0cc18965-c967-4b58-ac7f-38a443b4cd16', 'd0f62f51-5a51-443b-bf7b-18e3d5b191dc',
        'c2ebb179-0a2d-4e84-95be-2837be406716', '3ac76f3c-f560-4666-af61-c693e4cd3ad4', '484bbf93-bc67-4726-8b81-6c4ab608c861',
        '11b19a5f-4d61-4b5d-980c-98cf0c16906a', 'edecaeb7-53d3-411a-a00c-2ce6226fde50', 'a43cb746-fa63-4d39-87cd-43a81fbf3a8e',
        '5a3cc8a5-584c-4171-ae2c-97bdbc5047a1', '30778350-508b-4cbe-bff2-8882d0743aed', 'dc652dd4-c0b5-4036-bb70-f71cd9fd488a',
        '6e4dd7f4-2155-407f-922b-25aeb04a47b7', 'ba75d8cb-bfb9-4d2b-ac2f-2b5b8697188c', '10c621d6-1f99-4045-82a0-47d3ea107ddc',
        '7a6b6aeb-af0e-441f-97b5-d5db846bb045', '8e2cfb59-d31b-4346-b67a-b96a12ca04f5', '2661fcc0-9b3c-45bd-b60e-3c6351acabd1',
        'efcce7b2-e33a-4510-af56-09db1bfb5bd0', 'c643a6ba-a8a3-42de-afcc-fa92328397b7', '86c8d140-2b1d-41df-ba97-a959c54d2c19',
        '9826d43a-6be2-49ac-bfd7-fa2475f62985', '151f38af-a59a-42f8-9b2e-2df3fef3f658', '5b7b42ed-e41a-4e97-806a-6287ac918537',
        'd897ffe6-43a8-4ebc-9881-6097be7711e7', '95aa9a6b-b80a-4dcc-a0a4-228f53bc7959', '0330f00b-412e-44d5-b7cc-23bebd26fa88',
        '9742751d-d670-4310-97ab-a14973112470', '72df8c4b-f0ef-438d-9858-88053cb188c1', '05b4cb03-5576-4d79-9afa-1c6318d632ec',
        '6db73b00-b6f1-4f63-9fd0-49f518361ee2', '69df632e-d2aa-4005-a9e2-1c5e07eeebd9', '09aff9df-7c67-4810-8e13-90f8c9bd05a1',
        '3de1e810-7abe-4dd4-9663-e19270c37c52', 'afd8d475-9bd3-4740-a5be-293cd211b34d', 'de0a8389-d573-483c-811b-e7829bd58a54',
        '65b60f43-6a71-4c7b-805f-6c3f564c87bb', '8d1746ff-83c6-4675-acd7-01a2a654ec0a', '011866d3-76a6-4b5a-9c42-447e2d567892',
        '23d5ddd1-1fb0-4149-bf59-a2e7cd34213a', '1990750e-de0f-4789-9dcc-dae5b9b99173', 'b127b87c-600d-4f28-b74a-e6c33d27e42f',
        'bf0f6ceb-62fd-4a92-a7b0-29cf898b05e1', '103881e7-8415-4d6b-90c5-cef06f36b5b3', '8c558c8b-1a04-4c38-9f98-20cd5c8195a7',
        '5c025c8a-a032-494f-8204-dd92b1067448', '441fe9aa-79d7-4e27-8fb5-213c77c4f295', 'b630c1ad-f3a1-41c3-8e34-735a78dec9d1',
        '3200dbee-2685-48c7-a7dc-2e780853efda', '46583ddd-855b-4e6d-8711-d7b5a4fd26c1', 'a497e35d-fd84-4be2-a45d-f847962cd5f8',
        '295d5dc4-f4d4-4016-8fc0-4badd1b9c94e', 'f8934f7d-e1f6-462e-8a12-dc82c440bc90', '30fd755b-9f29-4330-b4d3-8eccc44ffade',
        'a9cf3c28-8b08-451a-830f-b737936a9579', '273b434f-74b7-4581-9d69-13f030b67313', 'fcf115d7-4246-4790-a89d-666f368b3356',
        'c1306680-d5e0-4629-939d-ee9e3f4c439b', 'a5cafc37-5ab6-40b2-8c1b-19089e724b1d', '34606fe8-3ba5-4778-a7f0-0275c1def3b8',
        '2d6ffd36-f33d-4a68-9868-53c7aa3f4011', '38e4e4d4-f801-42a0-8eac-a1a9a41a8a3e', 'bec0103f-bdf4-42b5-b04e-a44528b8c8d1',
        '1e01bc5e-dcdb-4ef1-94a6-f4e3a77613b8', '5debda2f-e5c1-447e-b025-d71252591074', '2d8872a4-f002-4294-9396-91d9cefabdb7',
        'ac4adfe9-a213-45bd-843b-f346c9891b2c', 'b59c4643-2b45-45e4-89f7-007ef1955c9f', 'f56f6f85-9aff-4f4e-bd84-8d763708e76f'
      ], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
    ),

    # KPI-AF-029: AI Response time (Railway)
    "1e226684-a836-468d-9929-b95bbf2f88dc": TestRunner_KPI_AF_029_Railway(
      test_id="1e226684-a836-468d-9929-b95bbf2f88dc", scenario_ids=['c5219c2e-c3b9-4e7a-aefc-b767a9b3005d'],
      benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
    ),

    # KPI-NF-045: Network Impact Propagation (Railway)
    "e075d4a7-5cda-4d3c-83ac-69a0db1d74dd": TestRunner_KPI_NF_045_Railway(
      test_id="e075d4a7-5cda-4d3c-83ac-69a0db1d74dd", scenario_ids=[
        'bb6302f1-0dc2-43ed-976b-4e5d3126006a', 'f84dcf0c-4bde-460b-9139-ea76e3694267', '89ea38d1-e42e-430e-8a72-f426f1cc0be7',
        'ac3d32bf-2694-4405-953b-01849e7923ef', '30286226-29a3-4aa6-8243-562b88967d76', '18276866-5a94-412b-b09c-9cac2ca5add0',
        '02e163b8-d8a3-44cb-9fb0-65501dfa35b7', 'ab2b11c8-66f4-47c3-9cd3-f765eb772dc7', 'f3ae4180-86f3-409a-a51e-c1deb7e005cd',
        '7a3ae3eb-b783-44a3-80d4-aa9cb0bd55fb', 'cff75f1a-8ea2-4f1d-b516-60dd0d625fe1', 'aa4fd74f-4680-405b-a184-c9392f9218e3',
        '01a82553-8d2c-4f84-94df-ccb9f3250734', '70316412-5480-44ca-9c2b-c51426b0390e', '60a6acda-9a1a-4a0a-8c04-75de02304713',
        'db614cef-8b86-467d-a638-64c25a91ec78', '43b053bb-5e9b-4538-a490-fee839344203', 'e01032e4-2047-455a-a329-175a40a8de24',
        '3b68eeb2-96f6-4a87-8a2f-5decaf3cb3f0', 'fdd89c15-3f8d-4381-9cd7-e8b773d06997', '1d8f2bda-38a4-41de-a614-291b9e4697e4',
        '7277b987-4cc2-4cb5-a308-bb226c832747', '62e20486-eb7f-49d9-a9dc-7aa00fdfefb0', 'ae7a8233-8a80-496a-a2b3-0afd9a28ebe6',
        '2b4b92d2-6871-4c20-ad58-11dc51718379', '86f360de-8c4d-44d0-b089-3259a91dc3ea', 'f4b1aaeb-a498-428e-8f8f-2ed07aee0641',
        'e5968696-5497-496b-8fe4-f40a837f7129', 'dec5cd4e-10b7-4a7e-a803-10e50badaaf5', '139b31bd-22e4-495c-8e21-5e6e34cd5a20',
        '9b603e03-3e2e-4366-8127-96307d3b2ce1', '54601145-edd9-469f-8180-245e26dff069', '34fa69f7-e0f7-4fd4-adf0-2ed8a47d6abc',
        '51081d92-2ff1-40a4-b557-38215c125051', 'fef8ec79-80da-4039-9484-6ec49a29263e', 'd9d80121-bf6c-41ee-bc18-dade0e853ada',
        '5832961b-942f-4d33-8614-c6dd4861ef46', 'e44e2b83-ad54-4e9c-a2ea-c23c1a249c54', '8ca33f45-3839-4ccc-aa88-146b41fee9f3',
        'e15fa3d0-0da3-4513-a5bd-6082806039a3', 'e28dc7e5-03ae-4687-ba37-c7ed5914c901', 'ef52b0f5-a147-4333-9817-fbd7e53143ee',
        '45b93b12-57cc-40ff-b277-82de8ceaec32', 'ae557fe8-2155-42b3-8d17-2e9de47dda4b', '3b2f7caf-2e32-4db9-8377-e01f50e436c2',
        '9ae1a2b5-fe89-4027-b1d8-8c3888862a5e', '9c221d41-fda7-409e-9ceb-a0f94018a92c', '7da98e75-8c84-4cfc-98f4-0fedf1aec08f',
        '7b42a1cc-ce70-4d9a-804f-ac9027a1ee48', '48caf228-64f6-4b03-ad20-5a34cf8dd2ee', '49cac9a9-1aac-4542-a01d-6483052bf02b',
        'b30319f8-8953-4433-80b6-5b80c9103bc5', '401a5b54-feb8-4eaf-92c3-426cb2f221ef', '06863bb7-48d5-4897-87c7-3328546efdef',
        '25f59eeb-3baf-4668-bdb1-2beb577fbf73', '2a631e96-a912-4b27-b82e-57ca3dd4aacf', 'a85cd328-09f9-4360-ae04-4479301b5987',
        'a35f5412-b565-4f24-9459-eb9ac1f7fe30', '2060f4fe-4f43-4095-b14f-a3c8ce312a42', '4e2e9ee7-26e1-4a2c-bc3c-93761a0ea43c',
        'f262ffb3-86b9-4db8-8657-f4a96915cb83', 'b022b575-103d-4ded-8da5-2e9a8f686da6', 'ba5308f8-7e12-4c6e-8e4f-42f6280f537c',
        'fec35ca6-d093-4081-8cf1-2f3b8f445bed', 'a410e586-219d-402f-9e34-6a1720ae46bc', 'a82beae8-9b1a-4a00-bea2-4891b56f3014',
        '14a20672-ea9c-4205-961a-4f2a3585eea2', 'e7f82820-1caf-4b4f-ae70-8ea4d95dcb0c', '989cc7e2-1d39-4334-8130-b704fd7c6c7b',
        '63b933c7-b63c-4ab7-b602-69aa5f91aec9', '05826982-694c-4ba3-817f-979d69942d36', 'dc280d8b-d3ca-4517-9d31-9e70e2f3aea8',
        '4223ef16-0a01-404d-8024-5a656203d3f8', '320b95b2-84d6-4827-ba37-0de57a1e6360', 'b9164bdc-9f5e-42b8-8379-7b6f454a3933',
        '59b4935d-cc5d-4792-a395-770bae030d2d', 'be970bda-465f-42b4-9223-c4ba742b24ea', '8df90815-5b7b-46c6-b388-546efbfa18a0',
        '10ee5c39-9ae9-4e58-bdc8-8a449887574d', '23ce8d72-6c32-45b4-a04e-34e029eb509e', 'f185083b-3f74-4221-b5c8-7b2e561ae2e4',
        'dc4da56a-bf2d-4351-b280-d418736844de', '02183d8b-2328-4467-81e9-97afde5618c9', 'd85cbc22-951e-4926-94f9-9c7a703b54eb',
        'd4b12805-017e-46d9-8fe7-220569a21477', '2f5a8e75-521e-4398-bfd2-e7ff7c9e0be4', '701c5b31-06d0-4e54-82b9-08e1612f1042',
        '6acbd77e-18a8-41c9-af81-1ff0ac1a9b0f', '20ef1912-26a4-48c1-ad1b-f08c95b144e2', '890c8007-a763-4a94-86d5-28d8c11c573f',
        'a7a37c14-b2a7-471c-9ed2-af90ee502d39', '848a39f3-e96b-4c41-83b8-78a2eb99403f', '5dbc518b-6a94-4e7f-b140-f99eb30af9b2',
        '626f428b-0928-48c8-8770-12de6e3b18ed', '6129f9a8-853c-459e-8ac2-aa7cbc65802e', '5869ddd8-fbb9-431e-beb7-64761220e3e3',
        'd238a30f-2f16-4e12-83a6-fcf779cd7115', 'e86883e7-53be-4b58-99cb-efd5f23cdb5b', '08c0e125-3ee8-45d4-94fc-84aa5d8c711d',
        'ac06a723-e801-470d-991d-b4411368ccfe', '566f099a-2957-4fc9-8e13-f9564311ba33', 'b990e1ab-5a11-4eba-9719-3b8b77846365',
        '9eb9657e-5c48-461a-a680-7fae151f6800', '2ff0136f-7995-4134-8e4e-9fd92da46ea8', 'fb08a40e-b5cc-4f8a-9b72-c9fe901fcb04',
        '1667a5c9-ae17-4628-ba55-74e34cb04332', '387dafe7-97ad-4b37-88e4-10785748e04d', '6735f34e-d3a1-41d5-86a0-f623099bf2cf',
        '2bb4384c-1119-4756-a2ad-fc2e5c00f952', '5a52729c-4f09-4f58-852d-48239e9ce217', '4c6e1a87-a083-46c1-b928-0c1d1763a9db',
        'c6ea3dff-9cda-471d-bf8b-a576b17036bb', '49257861-320d-4aa2-aa62-b9c4e1751791', 'be2823fd-5389-4415-9447-ed84006cee4c',
        '03ffcc01-b691-406f-ac53-5d13e9328175', '220bf75b-d271-4c46-bde2-67b313f10d13', 'eb2a0321-e6d4-49c7-8885-7e046018e12d',
        '119b0c6b-4ce6-4a7f-b49b-ec904577a182', '492c01f3-a7d4-46e5-b140-e82a22b13cdf', '45107995-4a41-4831-98c9-df7ed79734b1',
        'f3ee3bb9-3328-450a-98fb-63692042134f', 'd9f53db5-8cec-4ec4-8d62-d116cb198811', '0c2f1499-dbbe-43df-be32-f3e980ac1691',
        '6c90d7a4-980f-4e13-856c-13117e2edf82', '7a335c56-af9e-4562-a614-bfe84a75951f', '6267d830-501a-4646-823e-adab3403faf0',
        'f7509133-3083-4454-a963-95302eb66764', 'bb2b5fe3-92ed-4270-9235-dafbfffa2d03', '9b3fc207-3e88-4ef8-90b0-8ab1e77ea932',
        'bf4447a8-9e3e-4b8c-b9ea-4bc5bb009977', '42b7c577-c4ba-43e0-8ff9-d1839d86a06c', 'ae32ef91-586e-4dfd-b42b-e76002335794',
        'd271e605-1b68-4609-884c-0e6b2417980a', 'ef64aa0d-ba0d-4af4-abf3-22395713f0d7', '2f289e62-3b78-4580-90ed-c120947f70b5',
        '7dd80b01-99c3-4a89-9b4b-f4c878a6d996', '56e0d8cc-b7f0-4e9f-ae77-a7322cc9a2a5', 'ac523ef0-6a7a-470d-a944-795305899de9',
        'f9b8c0b3-0968-4324-9b92-35063c49def2', '8540245e-c841-4caa-aeb7-5dd6955bc43d', 'b7a9d4ba-c51f-452d-9128-3f923906ec18',
        '84db3ada-efe4-4cf7-bf05-3f14bbe2c668', 'c7864f2d-ae67-42a9-bddb-59b6933b0c1c', '4a9d8df0-d87b-4e31-9762-739f03694828',
        '05eeb2ea-67fe-405e-b630-43f382dbf246', '87e0d07c-cd5f-4506-b112-619d298ce924', 'c3b92403-d342-4d2e-b107-ae4ab443798f',
        '90071fa0-a560-4c6e-b2ff-fd59588fbdb7', '97bbc19c-de0c-4deb-838d-5675d9525eb8', 'cb55a7a4-460a-48e6-a623-4ebbc88b7be7'
      ], benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8"
    ),
    #
    #     # KPI-HS-018: Human control/autonomy over the process (Railway)
    #     "6a896809-dad8-4248-a2f3-d54373953fe6": TestRunner_KPI_HS_018_Railway(
    #         test_id="6a896809-dad8-4248-a2f3-d54373953fe6", scenario_ids=['dddb0b01-6bab-408e-9b12-4d7ab8e3b542'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
    #     ),
    #
    #     # KPI-IS-041: Impact on workload (Railway)
    #     "0d901459-427d-43ea-9c97-3815eaa52bf6": TestRunner_KPI_IS_041_Railway(
    #         test_id="0d901459-427d-43ea-9c97-3815eaa52bf6", scenario_ids=['e967a3b5-6d9b-41c2-9897-4e40a186e879'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
    #     ),
    #
    #     # KPI-AF-050: AI-Agent Scalability Training (Railway)
    #     "dff7e358-ff14-45e7-bc22-aac2b50500f3": TestRunner_KPI_AF_050_Railway(
    #         test_id="dff7e358-ff14-45e7-bc22-aac2b50500f3", scenario_ids=['d7cea956-6803-488c-b402-079d13b892c6'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
    #     ),
    #
    #     # KPI-AF-051: AI-Agent Scalability Testing (Railway)
    #     "b2e91a79-1390-414f-bf5d-8a6fd93c6080": TestRunner_KPI_AF_051_Railway(
    #         test_id="b2e91a79-1390-414f-bf5d-8a6fd93c6080", scenario_ids=['1f2b1af1-dc36-49ae-9322-e61656951545'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
    #     ),
    #
    #     # KPI-DF-052: Domain shift adaptation time (Railway)
    #     "0c3b9843-1940-45ef-943a-dc13ec1d090a": TestRunner_KPI_DF_052_Railway(
    #         test_id="0c3b9843-1940-45ef-943a-dc13ec1d090a", scenario_ids=['adb8b61b-6203-436d-9b40-8d4f95af7f43'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
    #     # KPI-DF-053: Domain shift generalization gap (Railway)
    #     "fba17d1f-2eb7-4c06-8f77-5da14e7b875a": TestRunner_KPI_DF_053_Railway(
    #         test_id="fba17d1f-2eb7-4c06-8f77-5da14e7b875a", scenario_ids=['d741cc0b-7dd9-4c2e-b23c-805b622c3878'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
    #     # KPI-DF-054: Domain shift out of domain detection accuracy (Railway)
    #     "69259fb3-bba8-4e62-9103-552b439ebaf4": TestRunner_KPI_DF_054_Railway(
    #         test_id="69259fb3-bba8-4e62-9103-552b439ebaf4", scenario_ids=['8d292cff-0409-4ec7-83fb-3034c3a4106c'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
    #     # KPI-DF-055: Domain shift policy robustness (Railway)
    #     "dce1c0e9-f2ae-4faa-a6b3-76d037c082d0": TestRunner_KPI_DF_055_Railway(
    #         test_id="dce1c0e9-f2ae-4faa-a6b3-76d037c082d0", scenario_ids=['4b42d9b8-9f27-4247-8bef-47ad3ce3a978'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
    #     # KPI-DF-056: Domain shift robustness to domain parameters (Railway)
    #     "03e1fcaf-a933-4ad1-abf2-5541574298a2": TestRunner_KPI_DF_056_Railway(
    #         test_id="03e1fcaf-a933-4ad1-abf2-5541574298a2", scenario_ids=['2238fa19-e337-49ce-a3b3-08abf85f1453'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
    #     # KPI-DF-057: Domain shift success rate drop (Railway)
    #     "d795c7bd-f8be-43c2-ae6e-a931194d3fc5": TestRunner_KPI_DF_057_Railway(
    #         test_id="d795c7bd-f8be-43c2-ae6e-a931194d3fc5", scenario_ids=['5d017ca4-f062-44c9-821c-31f85928903a'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
    #     # KPI-DF-090: Domain shift forgetting rate (Railway)
    #     "511f2ab0-da90-4d55-a23f-af5eda0baf7d": TestRunner_KPI_DF_090_Railway(
    #         test_id="511f2ab0-da90-4d55-a23f-af5eda0baf7d", scenario_ids=['cf7bb259-0ad4-4454-a9c8-eb8add0bec57'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
    #     ),
    #
        # KPI-RS-058: Robustness to operator input (Railway)
        "1cbf44c3-0c82-4f9e-9857-c7c1d96d3ab9": TestRunner_KPI_RS_058_Railway(
            test_id="1cbf44c3-0c82-4f9e-9857-c7c1d96d3ab9", scenario_ids=[
                "5a60713d-01f2-4d32-9867-21904629e254", "0db72a40-43e8-477b-89b3-a7bd1224660a", "7def3118-2e9c-4de7-8d61-f0e76fbeee5d", 
                "3ae60635-6995-4fb1-8309-61fded3d6fd8", "eeef8445-723d-4740-b89f-4dbaf75f9ae6", "94af1ed1-3686-4a9e-99f5-3a7ad908f125", 
                "8250d0e2-700e-4051-85c3-a8d0d95a5f0f", "c58759a7-a64a-4cbf-970b-948bae0c2254", "f94f517f-c0a4-4415-b726-186cdc75f9c6", 
                "c0e2c3e0-c171-48dd-a312-5de070e3f937", "6fc5f67a-40fa-45ce-819e-35a85e08e560", "66bce513-502c-43b4-a155-8a16c410a7c6", 
                "eff645bf-7ea8-490d-ae8a-ebb0d16a774c", "8397e6d6-babc-469b-a239-7eabcbd510da", "c359f13c-d222-4b04-ad0a-2bb30fb9da5f", 
                "97203764-6717-4ca6-bae9-c35c4eb38206", "adc4bf52-096c-4369-a85f-c9bf4b86bc64", "72f93d48-ecef-4bf7-9d97-cb008b47e566", 
                "b470667b-d9c9-4af4-b64e-c32102c34387", "4aa9e1b8-8669-466e-b4b9-c7db2a098bec", "8b308495-7ea6-4ddc-acb4-56eb5b3aec12", 
                "a8f69dc4-04a1-434a-ad97-27c745561b6a", "8b244f56-50e1-411a-a7d8-a2b89dfab26e", "8e6419c1-6470-4272-9c4b-43d9fe19dd3d", 
                "ec503b6e-3682-4dcd-9dc7-b194b67283d9", "74fd9eab-d2e5-4222-8656-81fc2dde7c21", "c16e54c1-33b2-45b8-95b0-33cc4f5400d5", 
                "c80effec-27b8-4103-b726-344a85f35407", "9bec9335-3dd5-4d88-b2ac-c5d711bcab36", "4a067d3c-75e6-4e91-a42d-cdf291016674", 
                "a5dff3c8-902e-4cb9-8466-d277d0ed4d67", "aca25feb-6254-40b3-8d40-3c805797c69b", "deb21442-0f94-4ff3-b78d-8d418415d646", 
                "6cf2cc89-d30e-4063-bced-051f3cdae92f", "84bcbff5-346f-452c-87ab-08ceff6364f2", "9acbe68e-2a45-420b-a142-34996dbcfb83", 
                "42786e4c-c80e-40f5-8237-bafc5f39979d", "242a6240-b62c-48b4-a264-b6737e893fa5", "3c38a1d3-2340-43ed-ac0b-4b76c6588b92", 
                "b89daede-405b-411a-a02b-ee32d7c9d020", "9e0aac9e-ddf9-4575-bf1c-d08a923e15fa", "befa97fb-2a74-4f2e-91c8-ea2879d08dcf", 
                "0cc18965-c967-4b58-ac7f-38a443b4cd16", "d0f62f51-5a51-443b-bf7b-18e3d5b191dc", "c2ebb179-0a2d-4e84-95be-2837be406716", 
                "3ac76f3c-f560-4666-af61-c693e4cd3ad4", "484bbf93-bc67-4726-8b81-6c4ab608c861", "11b19a5f-4d61-4b5d-980c-98cf0c16906a", 
                "edecaeb7-53d3-411a-a00c-2ce6226fde50", "a43cb746-fa63-4d39-87cd-43a81fbf3a8e"
                ], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
    #
    #     # KPI-DF-069: Drop-off in reward (Railway)
    #     "a94c858e-4bc3-4d67-bd78-5c81506e39f7": TestRunner_KPI_DF_069_Railway(
    #         test_id="a94c858e-4bc3-4d67-bd78-5c81506e39f7", scenario_ids=['74dd5830-6e59-423f-89f4-b050319db14e'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
    #     ),
    #
    #     # KPI-FF-070: Frequency changed output AI agent (Railway)
    #     "5abadf6b-991c-4d37-810f-f77bb71d490d": TestRunner_KPI_FF_070_Railway(
    #         test_id="5abadf6b-991c-4d37-810f-f77bb71d490d", scenario_ids=['ffcadd8d-207a-49af-8b09-54e922642f01'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
    #     ),
    #
    #     # KPI-SF-071: Severity of changed output AI agent (Railway)
    #     "dce32e78-e827-4994-a0a2-06feee2528cc": TestRunner_KPI_SF_071_Railway(
    #         test_id="dce32e78-e827-4994-a0a2-06feee2528cc", scenario_ids=['588ae37c-f583-47df-9154-ca12c9ac134a'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
    #     ),
    #
    #     # KPI-SF-072: Steps survived with perturbations (Railway)
    #     "e5206c56-75a0-41fa-9db3-bec66359337e": TestRunner_KPI_SF_072_Railway(
    #         test_id="e5206c56-75a0-41fa-9db3-bec66359337e", scenario_ids=['8011c7bd-6082-4653-8d9d-887d23f1ec5c'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
    #     ),
    #
    #     # KPI-VF-073: Vulnerability to perturbation (Railway)
    #     "0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc": TestRunner_KPI_VF_073_Railway(
    #         test_id="0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc", scenario_ids=['47a11418-fad5-4d55-a637-6b90a8351500'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
    #     ),
    #
    #     # KPI-RF-078: Reward per action (Railway)
    #     "8ebc88f0-896c-4910-8997-a44d107e7eb7": TestRunner_KPI_RF_078_Railway(
    #         test_id="8ebc88f0-896c-4910-8997-a44d107e7eb7", scenario_ids=['dc8195e4-266d-4afb-ba60-2659f59acfa4'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
    #     ),
    #
    #     # KPI-AF-074: Area between reward curves (Railway)
    #     "707a1a4e-7073-432b-94fc-af4a5ee9f07d": TestRunner_KPI_AF_074_Railway(
    #         test_id="707a1a4e-7073-432b-94fc-af4a5ee9f07d", scenario_ids=['a9e3fbf7-b5d5-477d-a0c8-d23880237d2d'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
    #     ),
    #
    #     # KPI-DF-075: Degradation time (Railway)
    #     "2c4be118-6108-43b3-b09f-a4bee842167a": TestRunner_KPI_DF_075_Railway(
    #         test_id="2c4be118-6108-43b3-b09f-a4bee842167a", scenario_ids=['dc03b9f1-bfb2-44b8-b124-f7eede10e0a7'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
    #     ),
    #
    #     # KPI-RF-076: Restorative time (Railway)
    #     "2cac54e0-aaf3-4f22-8307-f23878c432f0": TestRunner_KPI_RF_076_Railway(
    #         test_id="2cac54e0-aaf3-4f22-8307-f23878c432f0", scenario_ids=['56160b90-a287-4dec-acc7-f40967d60fa0'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
    #     ),
    #
    #     # KPI-SF-077: Similarity state to unperturbed situation (Railway)
    #     "d432299f-dbee-46ba-9e15-77954086440a": TestRunner_KPI_SF_077_Railway(
    #         test_id="d432299f-dbee-46ba-9e15-77954086440a", scenario_ids=['367a8b12-1b87-42f1-9400-4ecf96d6b617'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
    #     ),
    #
    #     # KPI-RS-091: Reflection on operator trust  (Railway)
    #     "9e680ab9-c861-4dd7-a5ea-abbdc1a91088": TestRunner_KPI_RS_091_Railway(
    #         test_id="9e680ab9-c861-4dd7-a5ea-abbdc1a91088", scenario_ids=['e9e537cc-a8d0-4864-9460-9098c72b269f'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
    #     # KPI-RS-092: Reflection on operator agency  (Railway)
    #     "8eb804ef-d122-4c59-a6b3-53f3c9664d2b": TestRunner_KPI_RS_092_Railway(
    #         test_id="8eb804ef-d122-4c59-a6b3-53f3c9664d2b", scenario_ids=['1ce89667-c4c2-483d-8ed1-c7add0a24b4b'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
    #     # KPI-RS-093: Reflection on operator de-skilling  (Railway)
    #     "8f5a1908-6319-4e2a-a43b-6528c7c5e92e": TestRunner_KPI_RS_093_Railway(
    #         test_id="8f5a1908-6319-4e2a-a43b-6528c7c5e92e", scenario_ids=['e283be95-96da-4f69-8869-06ab82ee2c45'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
    #     # KPI-RS-094: Reflection on over-reliance  (Railway)
    #     "a6a54b1d-54bf-4b09-9887-706eed3e1a57": TestRunner_KPI_RS_094_Railway(
    #         test_id="a6a54b1d-54bf-4b09-9887-706eed3e1a57", scenario_ids=['fbe33158-fb85-492d-9015-62b95f4658e8'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
    #     # KPI-RS-095: Reflection on additional training  (Railway)
    #     "0d898adb-7378-4ddd-8afa-a7136f13e3fa": TestRunner_KPI_RS_095_Railway(
    #         test_id="0d898adb-7378-4ddd-8afa-a7136f13e3fa", scenario_ids=['4a6d19a8-0d27-42d8-97af-9197987e64ce'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
    #     # KPI-RS-096: Reflection on biases  (Railway)
    #     "be770a02-c8be-4774-97b6-bc15e91b2fd8": TestRunner_KPI_RS_096_Railway(
    #         test_id="be770a02-c8be-4774-97b6-bc15e91b2fd8", scenario_ids=['b646c5dd-20e5-4628-984c-0b6ab3156538'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
    #     # KPI-PS-097: Predicted long-term adoption  (Railway)
    #     "2c71e014-3994-4413-a469-e27b0d1e121d": TestRunner_KPI_PS_097_Railway(
    #         test_id="2c71e014-3994-4413-a469-e27b0d1e121d", scenario_ids=['691ab2fd-9827-45f9-8d03-0878c1005144'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
    #     ),
    #
  }
)
