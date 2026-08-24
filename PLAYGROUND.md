# Playground for ai4realnet-int.flatland.cloud

3 [AIREALNET use cases](https://ai4realnet.eu/use-cases/)

## Power Grid

```sql
INSERT INTO field_definitions
  (id, key, description, agg_func, agg_weights)
  VALUES
  ('f270ab4e-ec52-469b-aa47-e9ea51810f04', 'normalized_reward', 'Scenario score (raw values)', NULL, NULL),
  ('cccf8ece-e0b3-4758-858e-4e178f702ea6', 'normalized_reward', 'Test score (NANSUM of scenario scores)', 'NANSUM', NULL),
  ('286aa338-435b-47ca-ab26-4d0cbb2b6e07', 'normalized_reward', 'Benchmark score (NANSUM of test scores)', 'NANSUM', NULL),
  ('4a1e85d5-9f75-494d-b449-fa000447abb2', 'percentage_complete', 'Scenario score (raw values)', NULL, NULL),
  ('227abd29-5075-4f66-a905-0c45279b2839', 'percentage_complete', 'Test score (NANMEAN of scenario scores)', 'NANMEAN', NULL),
  ('b611bf09-c7d5-419a-a9f8-bb6fd0dd6477', 'percentage_complete', 'Benchmark score (NANMEAN of test scores)', 'NANMEAN', NULL);

INSERT INTO scenario_definitions
  (id, name, description, field_ids)
  VALUES
  ('a11501c4-a28f-4b57-8d29-e62c9db35bc9', 'Test 0 Level 0', 'Test 0 Level 0', array['f270ab4e-ec52-469b-aa47-e9ea51810f04','4a1e85d5-9f75-494d-b449-fa000447abb2']::uuid[]),
('d25f9677-3131-4d03-91c1-82624ca94cc6', 'Test 0 Level 1', 'Test 0 Level 1', array['f270ab4e-ec52-469b-aa47-e9ea51810f04','4a1e85d5-9f75-494d-b449-fa000447abb2']::uuid[]),
('6c7b4616-43df-4858-84a1-630c89c04897', 'Test 0 Level 2', 'Test 0 Level 2', array['f270ab4e-ec52-469b-aa47-e9ea51810f04','4a1e85d5-9f75-494d-b449-fa000447abb2']::uuid[]),
('6eb92a98-3e79-4cc6-b078-c077500eeb81', 'Test 1 Level 0', 'Test 1 Level 0', array['f270ab4e-ec52-469b-aa47-e9ea51810f04','4a1e85d5-9f75-494d-b449-fa000447abb2']::uuid[]),
('a763693a-2d48-47b6-bfbc-1d7e3422abb0', 'Test 1 Level 1', 'Test 1 Level 1', array['f270ab4e-ec52-469b-aa47-e9ea51810f04','4a1e85d5-9f75-494d-b449-fa000447abb2']::uuid[]),
('dc281d3c-162a-4c0f-8056-ef94aa8299da', 'Test 1 Level 2', 'Test 1 Level 2', array['f270ab4e-ec52-469b-aa47-e9ea51810f04','4a1e85d5-9f75-494d-b449-fa000447abb2']::uuid[]);
INSERT INTO test_definitions
  (id, name, description, field_ids, scenario_ids, loop)
VALUES
('871f3eef-2bf4-4c04-ae6e-b6992581736a', 'Test 0', 'lorem ipsum', array['cccf8ece-e0b3-4758-858e-4e178f702ea6','227abd29-5075-4f66-a905-0c45279b2839']::uuid[], array['a11501c4-a28f-4b57-8d29-e62c9db35bc9', 'd25f9677-3131-4d03-91c1-82624ca94cc6', '6c7b4616-43df-4858-84a1-630c89c04897']::uuid[], 'CLOSED'),
('dc2f1e4c-043a-41f3-a457-7288d38d28da', 'Test 1', 'lorem ipsum', array['cccf8ece-e0b3-4758-858e-4e178f702ea6','227abd29-5075-4f66-a905-0c45279b2839']::uuid[], array['6eb92a98-3e79-4cc6-b078-c077500eeb81', 'a763693a-2d48-47b6-bfbc-1d7e3422abb0', 'dc281d3c-162a-4c0f-8056-ef94aa8299da']::uuid[], 'CLOSED');
INSERT INTO benchmark_definitions
  (id, name, description, field_ids, test_ids)
VALUES
  ('68d38d65-790b-4464-8fea-95d70a7a7de5', 'Playground Electricity Network', 'Playground Electricity Network', array['286aa338-435b-47ca-ab26-4d0cbb2b6e07','b611bf09-c7d5-419a-a9f8-bb6fd0dd6477']::uuid[], array['871f3eef-2bf4-4c04-ae6e-b6992581736a', 'dc2f1e4c-043a-41f3-a457-7288d38d28da']::uuid[]);
```

## Air Traffic Management

```sql
INSERT INTO field_definitions
  (id, key, description, agg_func, agg_weights)
  VALUES
  ('680d0569-8987-4d3a-84ba-2d393d833951', 'normalized_reward', 'Scenario score (raw values)', NULL, NULL),
  ('85734014-557d-489b-9c0b-b0729d492a7c', 'normalized_reward', 'Test score (NANSUM of scenario scores)', 'NANSUM', NULL),
  ('38f82a6c-0218-4ae6-8aab-7c1fe7f749ce', 'normalized_reward', 'Benchmark score (NANSUM of test scores)', 'NANSUM', NULL),
  ('e2a0afcc-69f4-4e64-9c5f-0f708049a7a8', 'percentage_complete', 'Scenario score (raw values)', NULL, NULL),
  ('05868dfa-9bfc-46c6-8ed2-edb5aff3d13c', 'percentage_complete', 'Test score (NANMEAN of scenario scores)', 'NANMEAN', NULL),
  ('22441689-e6a5-41c4-bc31-9dc27257f5eb', 'percentage_complete', 'Benchmark score (NANMEAN of test scores)', 'NANMEAN', NULL);

INSERT INTO scenario_definitions
  (id, name, description, field_ids)
  VALUES
  ('210f4cd4-ae68-4580-8401-a6f79e72f9f9', 'Test 0 Level 0', 'Test 0 Level 0', array['680d0569-8987-4d3a-84ba-2d393d833951','e2a0afcc-69f4-4e64-9c5f-0f708049a7a8']::uuid[]),
('78433dbc-ca7e-417c-8ea5-1ce50ec0911c', 'Test 0 Level 1', 'Test 0 Level 1', array['680d0569-8987-4d3a-84ba-2d393d833951','e2a0afcc-69f4-4e64-9c5f-0f708049a7a8']::uuid[]),
('698d67dd-ead5-4592-8194-bb364119e544', 'Test 0 Level 2', 'Test 0 Level 2', array['680d0569-8987-4d3a-84ba-2d393d833951','e2a0afcc-69f4-4e64-9c5f-0f708049a7a8']::uuid[]),
('ec720f11-8256-4c13-8520-cd277042350d', 'Test 1 Level 0', 'Test 1 Level 0', array['680d0569-8987-4d3a-84ba-2d393d833951','e2a0afcc-69f4-4e64-9c5f-0f708049a7a8']::uuid[]),
('9e46286c-418c-4167-998a-6e68ca66cd5d', 'Test 1 Level 1', 'Test 1 Level 1', array['680d0569-8987-4d3a-84ba-2d393d833951','e2a0afcc-69f4-4e64-9c5f-0f708049a7a8']::uuid[]),
('08fb6c0f-55c0-4934-9923-1c9336a114d2', 'Test 1 Level 2', 'Test 1 Level 2', array['680d0569-8987-4d3a-84ba-2d393d833951','e2a0afcc-69f4-4e64-9c5f-0f708049a7a8']::uuid[]);
INSERT INTO test_definitions
  (id, name, description, field_ids, scenario_ids, loop)
VALUES
('3a842fef-2190-4506-8fb5-b5cf1f06a1b9', 'Test 0', 'lorem ipsum', array['85734014-557d-489b-9c0b-b0729d492a7c','05868dfa-9bfc-46c6-8ed2-edb5aff3d13c']::uuid[], array['210f4cd4-ae68-4580-8401-a6f79e72f9f9', '78433dbc-ca7e-417c-8ea5-1ce50ec0911c', '698d67dd-ead5-4592-8194-bb364119e544']::uuid[], 'CLOSED'),
('ae913225-68f3-43b7-8308-0ca29a722565', 'Test 1', 'lorem ipsum', array['85734014-557d-489b-9c0b-b0729d492a7c','05868dfa-9bfc-46c6-8ed2-edb5aff3d13c']::uuid[], array['ec720f11-8256-4c13-8520-cd277042350d', '9e46286c-418c-4167-998a-6e68ca66cd5d', '08fb6c0f-55c0-4934-9923-1c9336a114d2']::uuid[], 'CLOSED');
INSERT INTO benchmark_definitions
  (id, name, description, field_ids, test_ids)
VALUES
  ('a4ac9d1b-7944-42a9-8134-77b7fba9f99c', 'Playground Air Traffic Management', 'Playground Air Traffic Management', array['38f82a6c-0218-4ae6-8aab-7c1fe7f749ce','22441689-e6a5-41c4-bc31-9dc27257f5eb']::uuid[], array['3a842fef-2190-4506-8fb5-b5cf1f06a1b9', 'ae913225-68f3-43b7-8308-0ca29a722565']::uuid[]);
```

## Railway Network

```sql
INSERT INTO field_definitions
  (id, key, description, agg_func, agg_weights)
  VALUES
  ('d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06', 'normalized_reward', 'Scenario score (raw values)', NULL, NULL),
  ('816f906a-b6f4-456c-8126-66fdd16b74e1', 'normalized_reward', 'Test score (NANSUM of scenario scores)', 'NANSUM', NULL),
  ('d78ddc15-ab85-4b0e-b221-51a0427e1003', 'normalized_reward', 'Benchmark score (NANSUM of test scores)', 'NANSUM', NULL),
  ('41311b12-d31f-4885-9b00-f7d990f54b82', 'percentage_complete', 'Scenario score (raw values)', NULL, NULL),
  ('746de734-0bd3-4de8-b99c-891d1dae0289', 'percentage_complete', 'Test score (NANMEAN of scenario scores)', 'NANMEAN', NULL),
  ('4ed0b046-098e-4e05-a358-517956e0571d', 'percentage_complete', 'Benchmark score (NANMEAN of test scores)', 'NANMEAN', NULL);

INSERT INTO scenario_definitions
  (id, name, description, field_ids)
  VALUES
  ('2049c534-a9dd-4bd8-905b-fe99e91f006b', 'Test 0 Level 0', 'Test 0 Level 0', array['d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06','41311b12-d31f-4885-9b00-f7d990f54b82']::uuid[]),
('7177093e-2fe5-4f13-b89e-ba5eccfddd23', 'Test 0 Level 1', 'Test 0 Level 1', array['d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06','41311b12-d31f-4885-9b00-f7d990f54b82']::uuid[]),
('998992a8-3f6c-415a-ab92-642477e4e32a', 'Test 0 Level 2', 'Test 0 Level 2', array['d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06','41311b12-d31f-4885-9b00-f7d990f54b82']::uuid[]),
('94c9e34d-5fbe-4a88-b303-0f33d34881f3', 'Test 1 Level 0', 'Test 1 Level 0', array['d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06','41311b12-d31f-4885-9b00-f7d990f54b82']::uuid[]),
('adcf2fee-bec7-4aba-afa1-da05790cbf33', 'Test 1 Level 1', 'Test 1 Level 1', array['d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06','41311b12-d31f-4885-9b00-f7d990f54b82']::uuid[]),
('cad1ebd3-484a-46fa-9638-94d73cf94d03', 'Test 1 Level 2', 'Test 1 Level 2', array['d0af3179-1c0e-48b6-93e3-1fcd2c9bbf06','41311b12-d31f-4885-9b00-f7d990f54b82']::uuid[]);
INSERT INTO test_definitions
  (id, name, description, field_ids, scenario_ids, loop)
VALUES
('216bf91c-5d0b-4704-a7bf-d913ad9c4598', 'Test 0', 'lorem ipsum', array['816f906a-b6f4-456c-8126-66fdd16b74e1','746de734-0bd3-4de8-b99c-891d1dae0289']::uuid[], array['2049c534-a9dd-4bd8-905b-fe99e91f006b', '7177093e-2fe5-4f13-b89e-ba5eccfddd23', '998992a8-3f6c-415a-ab92-642477e4e32a']::uuid[], 'CLOSED'),
('7cb6745c-960b-4508-9eba-90e1202d1e70', 'Test 1', 'lorem ipsum', array['816f906a-b6f4-456c-8126-66fdd16b74e1','746de734-0bd3-4de8-b99c-891d1dae0289']::uuid[], array['94c9e34d-5fbe-4a88-b303-0f33d34881f3', 'adcf2fee-bec7-4aba-afa1-da05790cbf33', 'cad1ebd3-484a-46fa-9638-94d73cf94d03']::uuid[], 'CLOSED');
INSERT INTO benchmark_definitions
  (id, name, description, field_ids, test_ids)
VALUES
  ('931e7cdf-a580-43f3-9d73-a54d07be13ac', 'Playground Railway', 'Playground Railway', array['d78ddc15-ab85-4b0e-b221-51a0427e1003','4ed0b046-098e-4e05-a358-517956e0571d']::uuid[], array['216bf91c-5d0b-4704-a7bf-d913ad9c4598', '7cb6745c-960b-4508-9eba-90e1202d1e70']::uuid[]);



INSERT INTO field_definitions
  (id, key, description, agg_func, agg_weights)
  VALUES
  ('2e6e2e91-ec2d-4d51-969a-54b2666ac377', 'normalized_reward', 'Scenario score (raw values)', NULL, NULL),
  ('ab5b8fec-8302-431e-be61-900a016a9a9c', 'normalized_reward', 'Test score (NANSUM of scenario scores)', 'NANSUM', NULL),
  ('77f6c296-52b8-44d4-b634-fe8d175c9038', 'normalized_reward', 'Benchmark score (NANSUM of test scores)', 'NANSUM', NULL),
  ('8c07bec9-fb94-4a0c-804e-472c5dc302de', 'percentage_complete', 'Scenario score (raw values)', NULL, NULL),
  ('762cfa8a-e557-4367-8390-a54814741c94', 'percentage_complete', 'Test score (NANMEAN of scenario scores)', 'NANMEAN', NULL),
  ('d83217a3-126e-487a-b5be-fc5be863cd50', 'percentage_complete', 'Benchmark score (NANMEAN of test scores)', 'NANMEAN', NULL);

INSERT INTO scenario_definitions
  (id, name, description, field_ids)
  VALUES
  ('cf8e0a9b-14af-43e1-b1fc-e43bf3aaddd7', 'Test 0 Level 0', 'Test 0 Level 0', array['2e6e2e91-ec2d-4d51-969a-54b2666ac377','8c07bec9-fb94-4a0c-804e-472c5dc302de']::uuid[]);
INSERT INTO test_definitions
  (id, name, description, field_ids, scenario_ids, loop)
VALUES
('c4c70f8a-679c-4044-a9d4-5e0ce0780a0f', 'Test 0', 'lorem ipsum', array['ab5b8fec-8302-431e-be61-900a016a9a9c','762cfa8a-e557-4367-8390-a54814741c94']::uuid[], array['cf8e0a9b-14af-43e1-b1fc-e43bf3aaddd7']::uuid[], 'INTERACTIVE');
INSERT INTO benchmark_definitions
  (id, name, description, field_ids, test_ids)
VALUES
  ('734144d1-c88c-4371-9dcf-72dd5dfe058e', 'Playground Railway (interactive)', 'Playground Railway (interactive)', array['77f6c296-52b8-44d4-b634-fe8d175c9038','d83217a3-126e-487a-b5be-fc5be863cd50']::uuid[], array['c4c70f8a-679c-4044-a9d4-5e0ce0780a0f']::uuid[]);
```
