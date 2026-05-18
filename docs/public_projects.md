# Public Java Project Execution Notes

This file records the public GitHub Java execution runs used while constructing
SAW-Bench-CS. The 105-project Java diversity set is the source of the released
403-warning v23 artifact. Earlier smoke, 20-project, and 50-project Maven runs
are retained below as construction provenance rather than as the current paper
benchmark.

## Toolchain

- Java: OpenJDK 17 from Homebrew (`/opt/homebrew/opt/openjdk@17`)
- Maven: Homebrew `maven`
- SpotBugs: Homebrew `spotbugs`
- Python: project-local `.venv`

Use this environment prefix for local commands:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH
```

## Smoke Project Set

| Split | Project | GitHub | Revision | Build |
| --- | --- | --- | --- | --- |
| train | Apache Commons Lang | [apache/commons-lang](https://github.com/apache/commons-lang) | `rel/commons-lang-3.14.0` | Maven |
| validation | Apache Commons IO | [apache/commons-io](https://github.com/apache/commons-io) | `rel/commons-io-2.15.1` | Maven |
| test | jsoup | [jhy/jsoup](https://github.com/jhy/jsoup) | `jsoup-1.17.2` | Maven |

Config files:

- `configs/projects_public_maven_smoke.yaml`
- `configs/splits_public_maven_smoke.yaml`

## Execution

Build an unlabeled source-derived JSONL:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH \
.venv/bin/python scripts/build_dataset.py \
  --config configs/projects_public_maven_smoke.yaml \
  --splits configs/splits_public_maven_smoke.yaml \
  --work-root data/public_maven_smoke/build \
  --out data/public_maven_smoke/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled \
  --drop-short-candidates
```

Validate the unlabeled records:

```bash
.venv/bin/python scripts/validate_dataset.py \
  data/public_maven_smoke/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled
```

The resulting file is not a paper-evaluable benchmark because it lacks
relevance labels. The released paper benchmark is built by merging the tracked
A/B pass files with `scripts/create_labeled_artifact_from_passes.py`.

`--drop-short-candidates` keeps the smoke output schema-valid by removing
warnings for which the lightweight extractor found fewer than five candidate
snippets.

## Local Run Record

Executed on 2026-04-28 with the toolchain above:

- Repositories cloned and pinned successfully.
- Maven builds completed for all three projects.
- SpotBugs reports were written under `data/public_maven_smoke/build/spotbugs/`.
- SpotBugs reported missing optional jsoup annotation classes:
  `org.jspecify.annotations.Nullable` and `org.jspecify.annotations.NullMarked`.
- Output JSONL: `data/public_maven_smoke/saw_bench_cs_unlabeled.jsonl`.
- Validation command passed with `--allow-unlabeled`.

Observed output counts:

| Project | Split | Warnings | Candidate snippets |
| --- | --- | ---: | ---: |
| commons-lang | train | 88 | 720 |
| commons-io | validation | 87 | 764 |
| jsoup | test | 53 | 469 |
| **Total** |  | **228** | **1,953** |

Average candidate snippets per warning: 8.57. Labels: 0.

## 20-Project Maven Set

The larger public set contains 20 GitHub-hosted Java/Maven repositories pinned
to Git HEAD SHAs queried on 2026-04-28. It is configured for source download,
project build, SpotBugs execution, candidate extraction, and unlabeled JSONL
generation.

Config files:

- `configs/projects_public_maven_20.yaml`
- `configs/splits_public_maven_20.yaml`

| Split | Project | GitHub | Revision | Build |
| --- | --- | --- | --- | --- |
| train | Apache Commons Lang | [apache/commons-lang](https://github.com/apache/commons-lang) | `0745c26dac9ac76086f10c302d252a71bf4a68c5` | Maven |
| train | Apache Commons IO | [apache/commons-io](https://github.com/apache/commons-io) | `2ccf3b6bee9e04a7a06d7c217b11329fbd1603e0` | Maven |
| train | Apache Commons CLI | [apache/commons-cli](https://github.com/apache/commons-cli) | `7be3807fa6fd4ad29241074f6ddeb23cfc677775` | Maven |
| train | Apache Commons Codec | [apache/commons-codec](https://github.com/apache/commons-codec) | `434789abd38decf069b2787696538580cfaf3359` | Maven |
| train | Apache Commons Collections | [apache/commons-collections](https://github.com/apache/commons-collections) | `3b157d054b51e9bf72bd7555fb4876e0fa003d66` | Maven |
| train | Apache Commons Compress | [apache/commons-compress](https://github.com/apache/commons-compress) | `426656d38254629acdec72764d4f90b95aaf4dbe` | Maven |
| train | Apache Commons CSV | [apache/commons-csv](https://github.com/apache/commons-csv) | `b6ab627e062ed80e65bcebe299f779717fca2e8a` | Maven |
| train | Apache Commons Configuration | [apache/commons-configuration](https://github.com/apache/commons-configuration) | `490c43e09283c70198388742171583054df9be48` | Maven |
| train | Apache Commons Net | [apache/commons-net](https://github.com/apache/commons-net) | `3af1604b0d3b6c1c69de7352f9d2fdf4d51d99fb` | Maven |
| train | Gson | [google/gson](https://github.com/google/gson) | `6bf8bf6cbf4d4f5f72f69262be890a5a5e28259d` | Maven |
| validation | Jackson Core | [FasterXML/jackson-core](https://github.com/FasterXML/jackson-core) | `2d66ad3c68ca93fe4d723dcda5a14e5fbd029ee9` | Maven |
| validation | Apache Commons DBCP | [apache/commons-dbcp](https://github.com/apache/commons-dbcp) | `d2877dd31a545c92b6e43a54e4f0c4e0e538f950` | Maven |
| validation | Apache Commons Validator | [apache/commons-validator](https://github.com/apache/commons-validator) | `fe87fdbff62c46b038438ba5eb72b71ca13aa6a9` | Maven |
| validation | Apache Commons Pool | [apache/commons-pool](https://github.com/apache/commons-pool) | `85ed7947f7f6614d827f38d3f43932ad460fb842` | Maven |
| validation | Apache Commons Text | [apache/commons-text](https://github.com/apache/commons-text) | `2c2122c5e6d94f6b1a7a095649b29df3aec7bf2d` | Maven |
| test | Jackson Databind | [FasterXML/jackson-databind](https://github.com/FasterXML/jackson-databind) | `8a70dbd6071f89f84634cd68cdecbbe2ed4e5d65` | Maven |
| test | jsoup | [jhy/jsoup](https://github.com/jhy/jsoup) | `ab88c1b8b69355155e9bf8d00b0713c81e054d4a` | Maven |
| test | Jackson Annotations | [FasterXML/jackson-annotations](https://github.com/FasterXML/jackson-annotations) | `85366adedaa535cfdc5cb40234fb3431be10874e` | Maven |
| *(failed)* | Apache Commons RNG | [apache/commons-rng](https://github.com/apache/commons-rng) | `86bf647f91ce1b54738bb143d989e697f5ae0dd5` | Maven |
| *(failed)* | Apache Commons Geometry | [apache/commons-geometry](https://github.com/apache/commons-geometry) | `0b9b4d7c6299e13c0f8aa653efbe4b2716c94c7c` | Maven |

Execution command:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH \
.venv/bin/python scripts/build_dataset.py \
  --config configs/projects_public_maven_20.yaml \
  --splits configs/splits_public_maven_20.yaml \
  --work-root data/public_maven_20/build \
  --out data/public_maven_20/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled \
  --drop-short-candidates \
  --continue-on-project-error
```

Validation command:

```bash
.venv/bin/python scripts/validate_dataset.py \
  data/public_maven_20/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled
```

## 20-Project Local Run Record

Executed on 2026-04-28/2026-04-29 with the toolchain above:

- All 20 repositories cloned and checked out at the pinned revisions.
- Maven build execution completed for the configured projects.
- SpotBugs/candidate extraction produced JSONL records for 17 projects.
- Output JSONL: `data/public_maven_20/saw_bench_cs_unlabeled.jsonl`.
- Validation command passed with `--allow-unlabeled`: 349 warnings, 0 errors,
  0 warnings.

Project-level execution failures recorded in
`data/public_maven_20/build/project_errors.txt`:

| Project | Failure |
| --- | --- |
| Gson | SpotBugs found no compiled classes under the checkout. |
| Apache Commons RNG | SpotBugs found no compiled classes under the checkout. |
| Apache Commons Geometry | SpotBugs found no compiled classes under the checkout. |

Observed output counts:

| Project | Split | Warnings |
| --- | --- | ---: |
| commons-cli | train | 4 |
| commons-codec | train | 3 |
| commons-collections | train | 33 |
| commons-compress | train | 89 |
| commons-configuration | train | 18 |
| commons-csv | train | 1 |
| commons-io | train | 26 |
| commons-lang | train | 17 |
| commons-net | train | 11 |
| jackson-core | validation | 31 |
| commons-dbcp | validation | 18 |
| commons-validator | validation | 12 |
| commons-pool | validation | 6 |
| commons-text | validation | 2 |
| jackson-databind | test | 63 |
| jsoup | test | 13 |
| jackson-annotations | test | 2 |
| **Total** |  | **349** |

Split totals: train 202, validation 69, test 78. Candidate snippets: 2,648.
Average candidate snippets per warning: 7.59.

**Note:** This run record covers Phase 1 (candidate extraction). The output file
`saw_bench_cs_unlabeled.jsonl` contains 0 labels by design. The released labeled
benchmark file `data/saw_bench_cs.jsonl` is regenerated from the tracked A/B
pass files by `scripts/create_labeled_artifact_from_passes.py`, which also
writes `annotation/annotator_passes.jsonl`.

## 50-Project Maven Diversity Set

The expanded public set contains 50 GitHub-hosted Java/Maven repositories pinned
to Git HEAD SHAs queried on 2026-04-29. This run was added to address project
concentration in the test split. The generated test split has 83 retained
warnings across 7 projects; the largest project contributes 27 warnings
(32.5%), and `jackson-databind` contributes 14 warnings (16.9%).

Config files:

- `configs/projects_public_maven_50.yaml`
- `configs/splits_public_maven_50.yaml`

| Split | Project | GitHub | Revision | Build |
| --- | --- | --- | --- | --- |
| train | commons-lang | [apache/commons-lang](https://github.com/apache/commons-lang) | `0745c26dac9ac76086f10c302d252a71bf4a68c5` | Maven fast |
| validation | commons-io | [apache/commons-io](https://github.com/apache/commons-io) | `2ccf3b6bee9e04a7a06d7c217b11329fbd1603e0` | Maven fast |
| test | jsoup | [jhy/jsoup](https://github.com/jhy/jsoup) | `ab88c1b8b69355155e9bf8d00b0713c81e054d4a` | Maven fast |
| train | commons-cli | [apache/commons-cli](https://github.com/apache/commons-cli) | `7be3807fa6fd4ad29241074f6ddeb23cfc677775` | Maven fast |
| train | commons-codec | [apache/commons-codec](https://github.com/apache/commons-codec) | `434789abd38decf069b2787696538580cfaf3359` | Maven fast |
| validation | commons-text | [apache/commons-text](https://github.com/apache/commons-text) | `2c2122c5e6d94f6b1a7a095649b29df3aec7bf2d` | Maven fast |
| test | commons-collections | [apache/commons-collections](https://github.com/apache/commons-collections) | `3b157d054b51e9bf72bd7555fb4876e0fa003d66` | Maven fast |
| test | commons-compress | [apache/commons-compress](https://github.com/apache/commons-compress) | `426656d38254629acdec72764d4f90b95aaf4dbe` | Maven fast |
| train | commons-csv | [apache/commons-csv](https://github.com/apache/commons-csv) | `b6ab627e062ed80e65bcebe299f779717fca2e8a` | Maven fast |
| train | commons-configuration | [apache/commons-configuration](https://github.com/apache/commons-configuration) | `490c43e09283c70198388742171583054df9be48` | Maven fast |
| validation | commons-validator | [apache/commons-validator](https://github.com/apache/commons-validator) | `fe87fdbff62c46b038438ba5eb72b71ca13aa6a9` | Maven fast |
| train | commons-net | [apache/commons-net](https://github.com/apache/commons-net) | `3af1604b0d3b6c1c69de7352f9d2fdf4d51d99fb` | Maven fast |
| train | gson | [google/gson](https://github.com/google/gson) | `6bf8bf6cbf4d4f5f72f69262be890a5a5e28259d` | Maven fast |
| validation | jackson-core | [FasterXML/jackson-core](https://github.com/FasterXML/jackson-core) | `2d66ad3c68ca93fe4d723dcda5a14e5fbd029ee9` | Maven fast |
| test | jackson-databind | [FasterXML/jackson-databind](https://github.com/FasterXML/jackson-databind) | `8a70dbd6071f89f84634cd68cdecbbe2ed4e5d65` | Maven fast |
| test | jackson-annotations | [FasterXML/jackson-annotations](https://github.com/FasterXML/jackson-annotations) | `85366adedaa535cfdc5cb40234fb3431be10874e` | Maven fast |
| validation | commons-pool | [apache/commons-pool](https://github.com/apache/commons-pool) | `85ed7947f7f6614d827f38d3f43932ad460fb842` | Maven fast |
| validation | commons-dbcp | [apache/commons-dbcp](https://github.com/apache/commons-dbcp) | `d2877dd31a545c92b6e43a54e4f0c4e0e538f950` | Maven fast |
| test | commons-rng | [apache/commons-rng](https://github.com/apache/commons-rng) | `86bf647f91ce1b54738bb143d989e697f5ae0dd5` | Maven fast |
| test | commons-geometry | [apache/commons-geometry](https://github.com/apache/commons-geometry) | `0b9b4d7c6299e13c0f8aa653efbe4b2716c94c7c` | Maven fast |
| train | commons-beanutils | [apache/commons-beanutils](https://github.com/apache/commons-beanutils) | `31eb40f2752f3208a1e6a18d96e760e78e88693d` | Maven fast |
| test | commons-jexl | [apache/commons-jexl](https://github.com/apache/commons-jexl) | `ac298d34e657eb6052e3d33d4922e42791d85120` | Maven fast |
| train | commons-vfs | [apache/commons-vfs](https://github.com/apache/commons-vfs) | `b1e2aaef0c54213c362207f7532ffd3014a98fe8` | Maven fast |
| train | commons-email | [apache/commons-email](https://github.com/apache/commons-email) | `beb92415f6cf8e2fed525187ec7dcebe4ba996f0` | Maven fast |
| train | commons-fileupload | [apache/commons-fileupload](https://github.com/apache/commons-fileupload) | `844aa562d36ae788d2dba44e6ec7722c58859eb8` | Maven fast |
| train | commons-imaging | [apache/commons-imaging](https://github.com/apache/commons-imaging) | `10cb083e09b17dbedd464d3250bc686bbd9607c8` | Maven fast |
| train | commons-exec | [apache/commons-exec](https://github.com/apache/commons-exec) | `2a987232a8d89d9570d4bb43a088fb81deeca314` | Maven fast |
| train | httpcomponents-client | [apache/httpcomponents-client](https://github.com/apache/httpcomponents-client) | `2ee445728efa81ea3cd6003faa6a9a4cca488015` | Maven fast |
| train | httpcomponents-core | [apache/httpcomponents-core](https://github.com/apache/httpcomponents-core) | `35e7d53cc89475bb51e956f92af61ab2393e68f6` | Maven fast |
| train | maven | [apache/maven](https://github.com/apache/maven) | `d27af1a895d7d430fc2805a4b898a0ff64a1cb75` | Maven fast |
| train | maven-resolver | [apache/maven-resolver](https://github.com/apache/maven-resolver) | `36df28226a5582bf3cd2bd4547312ad477014035` | Maven fast |
| validation | maven-surefire | [apache/maven-surefire](https://github.com/apache/maven-surefire) | `98cfb3d78b4485639f12af89b88aa5f9d8aef615` | Maven fast |
| validation | maven-shade-plugin | [apache/maven-shade-plugin](https://github.com/apache/maven-shade-plugin) | `c52adda94198bd88a19fe9eae509fc5d4801f801` | Maven fast |
| validation | junit4 | [junit-team/junit4](https://github.com/junit-team/junit4) | `300468b1efd48d76fac2f7bd6d576846dcbbf5ed` | Maven fast |
| validation | guava | [google/guava](https://github.com/google/guava) | `85512b085cbfb86bab7efcf2713fb5403551020f` | Maven fast |
| validation | truth | [google/truth](https://github.com/google/truth) | `eb7c034e25547f872fce7313f0f64dd48e1448c3` | Maven fast |
| validation | zxing | [zxing/zxing](https://github.com/zxing/zxing) | `bf77cd238d849a5602958e687cae56e6876de0c9` | Maven fast |
| validation | mybatis-3 | [mybatis/mybatis-3](https://github.com/mybatis/mybatis-3) | `5814ce2e4231eb51ea62355000aa5a51c155895d` | Maven fast |
| validation | HikariCP | [brettwooldridge/HikariCP](https://github.com/brettwooldridge/HikariCP) | `bba167f0a28905e8e63083cd7b5cbf479263271a` | Maven fast |
| validation | feign | [OpenFeign/feign](https://github.com/OpenFeign/feign) | `7b8e90d6c2f0eabcc66d19e38bc75a7228f399c5` | Maven fast |
| test | lettuce-core | [redis/lettuce](https://github.com/redis/lettuce) | `40f09234f15eb60ac45f84ecbe6147fd2d616472` | Maven fast |
| test | redisson | [redisson/redisson](https://github.com/redisson/redisson) | `c6e72cafa5c9efee27494003e45d7820b65f37de` | Maven fast |
| test | liquibase | [liquibase/liquibase](https://github.com/liquibase/liquibase) | `ff05483bcb450d0148bb7872ebd618cd6fcb7401` | Maven fast |
| test | flyway | [flyway/flyway](https://github.com/flyway/flyway) | `7c3ea883c0b78fd0aff1c69275cf3c0945eb6017` | Maven fast |
| test | pdfbox | [apache/pdfbox](https://github.com/apache/pdfbox) | `aba136447b22287e875496abbad3a14b5005459f` | Maven fast |
| test | tika | [apache/tika](https://github.com/apache/tika) | `6b538166612ef83aaefd838349bf8c59713faf40` | Maven fast |
| train | shiro | [apache/shiro](https://github.com/apache/shiro) | `4b2cd7dded4ca99d6a4a368b7ce5d4568b1a285b` | Maven fast |
| train | logging-log4j2 | [apache/logging-log4j2](https://github.com/apache/logging-log4j2) | `76c8bef88a5af15b09420445143e652f4696cab7` | Maven fast |
| train | slf4j | [qos-ch/slf4j](https://github.com/qos-ch/slf4j) | `d2073bece8310017914e1ff65d7107b2d4869c7d` | Maven fast |
| train | JsonPath | [json-path/JsonPath](https://github.com/json-path/JsonPath) | `62a4c9f0f65ba3f625aa0867d64c528ba72d09ec` | Maven fast |

Execution command:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH \
.venv/bin/python scripts/build_dataset.py \
  --config configs/projects_public_maven_50.yaml \
  --splits configs/splits_public_maven_50.yaml \
  --work-root data/public_maven_50/build \
  --out data/public_maven_50/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled \
  --drop-short-candidates \
  --continue-on-project-error
```

Validation command:

```bash
.venv/bin/python scripts/validate_dataset.py \
  data/public_maven_50/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled
```

## 50-Project Local Run Record

Executed on 2026-04-29 with the toolchain above:

- All 50 repositories were cloned and checked out at the pinned revisions.
- Maven fast build execution was attempted for all 50 projects.
- SpotBugs/candidate extraction produced JSONL records for 26 projects.
- Output JSONL: `data/public_maven_50/saw_bench_cs_unlabeled.jsonl`.
- Validation command passed with `--allow-unlabeled`: 414 warnings, 0 errors,
  0 warnings.
- Project-level build or SpotBugs misses are recorded in
  `data/public_maven_50/build/project_errors.txt`.

Observed split counts:

| Split | Warnings | Projects with retained warnings | Largest project share |
| --- | ---: | ---: | ---: |
| train | 247 | 9 | 21.1% |
| validation | 84 | 10 | 25.0% |
| test | 83 | 7 | 32.5% |
| **Total** | **414** | **26** |  |

Test split distribution:

| Project | Warnings | Share |
| --- | ---: | ---: |
| commons-compress | 27 | 32.5% |
| lettuce-core | 20 | 24.1% |
| jackson-databind | 14 | 16.9% |
| commons-jexl | 10 | 12.0% |
| commons-collections | 8 | 9.6% |
| jsoup | 3 | 3.6% |
| jackson-annotations | 1 | 1.2% |

## 105-Project Java Diversity Set

The 100+ public Java set contains 105 GitHub-hosted Java repositories pinned to
Git HEAD SHAs queried on 2026-04-29. It was added to stress project diversity
beyond the 50-project Maven set. The run attempts clone/checkout for all 105
projects, uses Maven fast builds where possible, and then runs SpotBugs over the
compiled bytecode that is available. Very large or non-standard builds are
allowed to fail so one oversized repository does not block the entire public
execution pass.

Config files:

- `configs/projects_public_java_100.yaml`
- `configs/splits_public_java_100.yaml`

Execution command used for the full build attempt:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH \
.venv/bin/python scripts/build_dataset.py \
  --config configs/projects_public_java_100.yaml \
  --splits configs/splits_public_java_100.yaml \
  --work-root data/public_java_100/build \
  --out data/public_java_100/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled \
  --drop-short-candidates \
  --continue-on-project-error
```

The full build pass was interrupted after ShardingSphere reached module 304 of
416; otherwise the run would spend most of its wall-clock budget in one reactor.
The final extraction pass reused the 105 pinned checkouts and already compiled
bytecode:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 \
PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH \
.venv/bin/python scripts/build_dataset.py \
  --config configs/projects_public_java_100.yaml \
  --splits configs/splits_public_java_100.yaml \
  --work-root data/public_java_100/build \
  --out data/public_java_100/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled \
  --drop-short-candidates \
  --continue-on-project-error \
  --skip-build
```

Validation command:

```bash
.venv/bin/python scripts/validate_dataset.py \
  data/public_java_100/saw_bench_cs_unlabeled.jsonl \
  --allow-unlabeled
```

## 105-Project Local Run Record

Executed on 2026-04-29 with the toolchain above:

- All 105 repositories were cloned and checked out at pinned revisions.
- Maven fast build execution was attempted before extraction; the largest reactor
  was interrupted and the extraction pass reused completed bytecode.
- SpotBugs/candidate extraction produced JSONL records for 33 projects.
- Output JSONL: `data/public_java_100/saw_bench_cs_unlabeled.jsonl`.
- Validation command passed with `--allow-unlabeled`: 403 warnings, 0 errors,
  0 warnings.
- Project-level build or SpotBugs misses are recorded in
  `docs/project_errors_public_java_100.txt` (72 project entries).

Observed split counts:

| Split | Warnings | Projects with retained warnings | Largest project | Largest project share |
| --- | ---: | ---: | --- | ---: |
| train | 249 | 13 | commons-scxml | 20.9% |
| validation | 77 | 11 | mybatis-3 | 20.8% |
| test | 77 | 9 | commons-compress | 26.0% |
| **Total** | **403** | **33** |  |  |

Test split distribution:

| Project | Warnings | Share |
| --- | ---: | ---: |
| commons-compress | 20 | 26.0% |
| lettuce-core | 17 | 22.1% |
| jackson-databind | 13 | 16.9% |
| commons-bcel | 8 | 10.4% |
| commons-collections | 7 | 9.1% |
| commons-jexl | 7 | 9.1% |
| jsoup | 3 | 3.9% |
| jackson-annotations | 1 | 1.3% |
| mybatis-spring | 1 | 1.3% |

The test split now has 77 retained warnings across
9 projects. `jackson-databind` contributes
13 test warnings
(16.9%), compared with 81% in the
reviewed 78-warning test artifact.

Configured project list:

| Split | Project | GitHub | Revision | Retained warnings | Status |
| --- | --- | --- | --- | ---: | --- |
| train | commons-lang | [apache/commons-lang](https://github.com/apache/commons-lang) | `0745c26dac9ac76086f10c302d252a71bf4a68c5` | 36 | retained |
| validation | commons-io | [apache/commons-io](https://github.com/apache/commons-io) | `2ccf3b6bee9e04a7a06d7c217b11329fbd1603e0` | 10 | retained |
| test | jsoup | [jhy/jsoup](https://github.com/jhy/jsoup) | `ab88c1b8b69355155e9bf8d00b0713c81e054d4a` | 3 | retained |
| train | commons-cli | [apache/commons-cli](https://github.com/apache/commons-cli) | `7be3807fa6fd4ad29241074f6ddeb23cfc677775` | 10 | retained |
| train | commons-codec | [apache/commons-codec](https://github.com/apache/commons-codec) | `434789abd38decf069b2787696538580cfaf3359` | 7 | retained |
| validation | commons-text | [apache/commons-text](https://github.com/apache/commons-text) | `2c2122c5e6d94f6b1a7a095649b29df3aec7bf2d` | 1 | retained |
| test | commons-collections | [apache/commons-collections](https://github.com/apache/commons-collections) | `3b157d054b51e9bf72bd7555fb4876e0fa003d66` | 7 | retained |
| test | commons-compress | [apache/commons-compress](https://github.com/apache/commons-compress) | `426656d38254629acdec72764d4f90b95aaf4dbe` | 20 | retained |
| train | commons-csv | [apache/commons-csv](https://github.com/apache/commons-csv) | `b6ab627e062ed80e65bcebe299f779717fca2e8a` | 1 | retained |
| train | commons-configuration | [apache/commons-configuration](https://github.com/apache/commons-configuration) | `490c43e09283c70198388742171583054df9be48` | 38 | retained |
| validation | commons-validator | [apache/commons-validator](https://github.com/apache/commons-validator) | `fe87fdbff62c46b038438ba5eb72b71ca13aa6a9` | 4 | retained |
| train | commons-net | [apache/commons-net](https://github.com/apache/commons-net) | `3af1604b0d3b6c1c69de7352f9d2fdf4d51d99fb` | 28 | retained |
| train | gson | [google/gson](https://github.com/google/gson) | `6bf8bf6cbf4d4f5f72f69262be890a5a5e28259d` | 0 | no retained warnings |
| validation | jackson-core | [FasterXML/jackson-core](https://github.com/FasterXML/jackson-core) | `2d66ad3c68ca93fe4d723dcda5a14e5fbd029ee9` | 12 | retained |
| test | jackson-databind | [FasterXML/jackson-databind](https://github.com/FasterXML/jackson-databind) | `8a70dbd6071f89f84634cd68cdecbbe2ed4e5d65` | 13 | retained |
| test | jackson-annotations | [FasterXML/jackson-annotations](https://github.com/FasterXML/jackson-annotations) | `85366adedaa535cfdc5cb40234fb3431be10874e` | 1 | retained |
| validation | commons-pool | [apache/commons-pool](https://github.com/apache/commons-pool) | `85ed7947f7f6614d827f38d3f43932ad460fb842` | 2 | retained |
| validation | commons-dbcp | [apache/commons-dbcp](https://github.com/apache/commons-dbcp) | `d2877dd31a545c92b6e43a54e4f0c4e0e538f950` | 5 | retained |
| test | commons-rng | [apache/commons-rng](https://github.com/apache/commons-rng) | `86bf647f91ce1b54738bb143d989e697f5ae0dd5` | 0 | no retained warnings |
| test | commons-geometry | [apache/commons-geometry](https://github.com/apache/commons-geometry) | `0b9b4d7c6299e13c0f8aa653efbe4b2716c94c7c` | 0 | no retained warnings |
| train | commons-beanutils | [apache/commons-beanutils](https://github.com/apache/commons-beanutils) | `31eb40f2752f3208a1e6a18d96e760e78e88693d` | 33 | retained |
| test | commons-jexl | [apache/commons-jexl](https://github.com/apache/commons-jexl) | `ac298d34e657eb6052e3d33d4922e42791d85120` | 7 | retained |
| train | commons-vfs | [apache/commons-vfs](https://github.com/apache/commons-vfs) | `b1e2aaef0c54213c362207f7532ffd3014a98fe8` | 0 | no retained warnings |
| train | commons-email | [apache/commons-email](https://github.com/apache/commons-email) | `beb92415f6cf8e2fed525187ec7dcebe4ba996f0` | 0 | no retained warnings |
| train | commons-fileupload | [apache/commons-fileupload](https://github.com/apache/commons-fileupload) | `844aa562d36ae788d2dba44e6ec7722c58859eb8` | 0 | no retained warnings |
| train | commons-imaging | [apache/commons-imaging](https://github.com/apache/commons-imaging) | `10cb083e09b17dbedd464d3250bc686bbd9607c8` | 18 | retained |
| train | commons-exec | [apache/commons-exec](https://github.com/apache/commons-exec) | `2a987232a8d89d9570d4bb43a088fb81deeca314` | 8 | retained |
| train | httpcomponents-client | [apache/httpcomponents-client](https://github.com/apache/httpcomponents-client) | `2ee445728efa81ea3cd6003faa6a9a4cca488015` | 0 | no retained warnings |
| train | httpcomponents-core | [apache/httpcomponents-core](https://github.com/apache/httpcomponents-core) | `35e7d53cc89475bb51e956f92af61ab2393e68f6` | 0 | no retained warnings |
| train | maven | [apache/maven](https://github.com/apache/maven) | `d27af1a895d7d430fc2805a4b898a0ff64a1cb75` | 0 | no retained warnings |
| train | maven-resolver | [apache/maven-resolver](https://github.com/apache/maven-resolver) | `36df28226a5582bf3cd2bd4547312ad477014035` | 0 | no retained warnings |
| validation | maven-surefire | [apache/maven-surefire](https://github.com/apache/maven-surefire) | `98cfb3d78b4485639f12af89b88aa5f9d8aef615` | 0 | no retained warnings |
| validation | maven-shade-plugin | [apache/maven-shade-plugin](https://github.com/apache/maven-shade-plugin) | `c52adda94198bd88a19fe9eae509fc5d4801f801` | 2 | retained |
| validation | junit4 | [junit-team/junit4](https://github.com/junit-team/junit4) | `300468b1efd48d76fac2f7bd6d576846dcbbf5ed` | 13 | retained |
| validation | guava | [google/guava](https://github.com/google/guava) | `85512b085cbfb86bab7efcf2713fb5403551020f` | 0 | no retained warnings |
| validation | truth | [google/truth](https://github.com/google/truth) | `eb7c034e25547f872fce7313f0f64dd48e1448c3` | 0 | no retained warnings |
| validation | zxing | [zxing/zxing](https://github.com/zxing/zxing) | `bf77cd238d849a5602958e687cae56e6876de0c9` | 0 | no retained warnings |
| validation | mybatis-3 | [mybatis/mybatis-3](https://github.com/mybatis/mybatis-3) | `5814ce2e4231eb51ea62355000aa5a51c155895d` | 16 | retained |
| validation | HikariCP | [brettwooldridge/HikariCP](https://github.com/brettwooldridge/HikariCP) | `bba167f0a28905e8e63083cd7b5cbf479263271a` | 4 | retained |
| validation | feign | [OpenFeign/feign](https://github.com/OpenFeign/feign) | `7b8e90d6c2f0eabcc66d19e38bc75a7228f399c5` | 0 | no retained warnings |
| test | lettuce-core | [redis/lettuce](https://github.com/redis/lettuce) | `40f09234f15eb60ac45f84ecbe6147fd2d616472` | 17 | retained |
| test | redisson | [redisson/redisson](https://github.com/redisson/redisson) | `c6e72cafa5c9efee27494003e45d7820b65f37de` | 0 | no retained warnings |
| test | liquibase | [liquibase/liquibase](https://github.com/liquibase/liquibase) | `ff05483bcb450d0148bb7872ebd618cd6fcb7401` | 0 | no retained warnings |
| test | flyway | [flyway/flyway](https://github.com/flyway/flyway) | `7c3ea883c0b78fd0aff1c69275cf3c0945eb6017` | 0 | no retained warnings |
| test | pdfbox | [apache/pdfbox](https://github.com/apache/pdfbox) | `aba136447b22287e875496abbad3a14b5005459f` | 0 | no retained warnings |
| test | tika | [apache/tika](https://github.com/apache/tika) | `6b538166612ef83aaefd838349bf8c59713faf40` | 0 | no retained warnings |
| train | shiro | [apache/shiro](https://github.com/apache/shiro) | `4b2cd7dded4ca99d6a4a368b7ce5d4568b1a285b` | 0 | no retained warnings |
| train | logging-log4j2 | [apache/logging-log4j2](https://github.com/apache/logging-log4j2) | `76c8bef88a5af15b09420445143e652f4696cab7` | 0 | no retained warnings |
| train | slf4j | [qos-ch/slf4j](https://github.com/qos-ch/slf4j) | `d2073bece8310017914e1ff65d7107b2d4869c7d` | 0 | no retained warnings |
| train | JsonPath | [json-path/JsonPath](https://github.com/json-path/JsonPath) | `62a4c9f0f65ba3f625aa0867d64c528ba72d09ec` | 0 | no retained warnings |
| test | commons-math | [apache/commons-math](https://github.com/apache/commons-math) | `67e3bacc001564eb58a45fc655b6aa78dc5a2bf0` | 0 | no retained warnings |
| validation | commons-numbers | [apache/commons-numbers](https://github.com/apache/commons-numbers) | `b620383d320788d26d876149b285f96ec2ddd2a4` | 0 | no retained warnings |
| train | commons-crypto | [apache/commons-crypto](https://github.com/apache/commons-crypto) | `d306bcdd375a5ed84989d20ea969c5389d0ebbfd` | 5 | retained |
| train | commons-dbutils | [apache/commons-dbutils](https://github.com/apache/commons-dbutils) | `723b6c2f45ddf3b823cfe7182a9ef3cb0a56abaf` | 11 | retained |
| train | commons-daemon | [apache/commons-daemon](https://github.com/apache/commons-daemon) | `dc4686f6481991c9c095bdc86a7dd165e54e4cfe` | 2 | retained |
| train | commons-scxml | [apache/commons-scxml](https://github.com/apache/commons-scxml) | `94823642231b983de89aae440bb3e5ede790b842` | 52 | retained |
| train | commons-weaver | [apache/commons-weaver](https://github.com/apache/commons-weaver) | `f40a7f3b6821bacbb5d5223b67c4e31efc44d81b` | 0 | no retained warnings |
| test | commons-bcel | [apache/commons-bcel](https://github.com/apache/commons-bcel) | `3c844f915c0c5698a9eccc48e68d90a92f87ecab` | 8 | retained |
| validation | commons-digester | [apache/commons-digester](https://github.com/apache/commons-digester) | `d238e78a8c2ece36fd30680428a4fde8417b326b` | 0 | no retained warnings |
| test | avro | [apache/avro](https://github.com/apache/avro) | `fed00117056cdc3dad424cf8442c2d38775e4658` | 0 | no retained warnings |
| test | parquet-java | [apache/parquet-java](https://github.com/apache/parquet-java) | `d5722f1e6bd7243d31a1b034027b89e1191ef603` | 0 | no retained warnings |
| validation | calcite | [apache/calcite](https://github.com/apache/calcite) | `b008df9a71cea5da44674c07e8599a0836ad1aad` | 0 | no retained warnings |
| test | curator | [apache/curator](https://github.com/apache/curator) | `90e1b7b59085293b34f711cf2e85c46b35eea1c9` | 0 | no retained warnings |
| validation | zookeeper | [apache/zookeeper](https://github.com/apache/zookeeper) | `afe8f08b6d369a4670a794443d5efcbfbb79ddc9` | 0 | no retained warnings |
| validation | bookkeeper | [apache/bookkeeper](https://github.com/apache/bookkeeper) | `46c842ca4a93ce295bf712bcd52287d52da1a0cf` | 0 | no retained warnings |
| validation | mina | [apache/mina](https://github.com/apache/mina) | `cd62e266374ef7a040de7a39c711c87a00cd498c` | 0 | no retained warnings |
| test | velocity-engine | [apache/velocity-engine](https://github.com/apache/velocity-engine) | `3c5449163f0d3edb85777eefa8f6f0cbd8cafcf5` | 0 | no retained warnings |
| test | struts | [apache/struts](https://github.com/apache/struts) | `f4c634928337f8114b03be2adcc8349e9d6c7358` | 0 | no retained warnings |
| test | wicket | [apache/wicket](https://github.com/apache/wicket) | `72470983f689c61e6a6c0b7388ef955f23bb1e16` | 0 | no retained warnings |
| test | poi | [apache/poi](https://github.com/apache/poi) | `e6a04b49211e23c704fcdbe524d99d2f4486b083` | 0 | no retained warnings |
| test | jena | [apache/jena](https://github.com/apache/jena) | `4ad0de7ae3890c504f206177c5a1dbcadda811cc` | 0 | no retained warnings |
| validation | commons-ognl | [apache/commons-ognl](https://github.com/apache/commons-ognl) | `1f4a423cf88efc74ea64a9ed1ec64171bb9e0b01` | 8 | retained |
| validation | commons-rdf | [apache/commons-rdf](https://github.com/apache/commons-rdf) | `ee77b23e7cbb6df5a6db36e831e0fd89306d5c0b` | 0 | no retained warnings |
| train | commons-build-plugin | [apache/commons-build-plugin](https://github.com/apache/commons-build-plugin) | `001c8b0a95cfa18ec060f28dddf5bd266429b62b` | 0 | no retained warnings |
| test | javapoet | [square/javapoet](https://github.com/square/javapoet) | `b9017a9503b76e11b4ad4c1a9f050e2d29112cb0` | 0 | no retained warnings |
| test | retrofit | [square/retrofit](https://github.com/square/retrofit) | `3e7bcf756797ab6ba035083b4e4d6db40a29edc4` | 0 | no retained warnings |
| train | okhttp | [square/okhttp](https://github.com/square/okhttp) | `e4ae9b02e913deec3a9da4c142064059436374fb` | 0 | no retained warnings |
| validation | assertj | [assertj/assertj](https://github.com/assertj/assertj) | `2c202d1d26f00bc0a6400ed2268cc45ffe38ed89` | 0 | no retained warnings |
| test | awaitility | [awaitility/awaitility](https://github.com/awaitility/awaitility) | `4fc23ccbd610ec16d14c2e7728e0593bffa6ac64` | 0 | no retained warnings |
| test | cucumber-jvm | [cucumber/cucumber-jvm](https://github.com/cucumber/cucumber-jvm) | `699f7a3a7fea6a33987c08a7ac83ea0f2a21942a` | 0 | no retained warnings |
| validation | junit5 | [junit-team/junit5](https://github.com/junit-team/junit5) | `f66dde69761d0495a0ab472b929068d12aa6539b` | 0 | no retained warnings |
| train | mockito | [mockito/mockito](https://github.com/mockito/mockito) | `32f06eb839d23958dc7a83c830e8a627e02e0dd0` | 0 | no retained warnings |
| train | wiremock | [wiremock/wiremock](https://github.com/wiremock/wiremock) | `12b40e6732bc3bdfab6f4a053b39097eee9e3f23` | 0 | no retained warnings |
| train | testcontainers-java | [testcontainers/testcontainers-java](https://github.com/testcontainers/testcontainers-java) | `afd28bb559b07f8da5aaa9e2bc21721ee8ee97af` | 0 | no retained warnings |
| test | netty | [netty/netty](https://github.com/netty/netty) | `27b6a1b97d0cfe1e892b0a50d0dd85f2137625df` | 0 | no retained warnings |
| train | zipkin | [openzipkin/zipkin](https://github.com/openzipkin/zipkin) | `878ce2a1fad54ca941d17fdcf2e1d924b148eb1f` | 0 | no retained warnings |
| validation | openapi-generator | [OpenAPITools/openapi-generator](https://github.com/OpenAPITools/openapi-generator) | `869f58f7bd5fc77d396db323409dfcc78990f644` | 0 | no retained warnings |
| validation | swagger-core | [swagger-api/swagger-core](https://github.com/swagger-api/swagger-core) | `3ebcde85ab6223ff57c5591afc0c89937e3cfff4` | 0 | no retained warnings |
| test | fastjson2 | [alibaba/fastjson2](https://github.com/alibaba/fastjson2) | `3697c2d37cd659d2a94543093d0d08cb4baf4d73` | 0 | no retained warnings |
| validation | druid | [alibaba/druid](https://github.com/alibaba/druid) | `2790bd782191a4824e1deea418ff92ecdcef41e9` | 0 | no retained warnings |
| train | easyexcel | [alibaba/easyexcel](https://github.com/alibaba/easyexcel) | `aae9c61ab603c04331333782eedd2896d7bc5386` | 0 | no retained warnings |
| test | mybatis-spring | [mybatis/spring](https://github.com/mybatis/spring) | `4a1b6e197955e205c39381ca32df4e2b13c4d908` | 1 | retained |
| validation | mybatis-spring-boot | [mybatis/mybatis-spring-boot](https://github.com/mybatis/mybatis-spring-boot) | `8933e413d51f88c8774a3812f1a0efe791fa558c` | 0 | no retained warnings |
| test | dubbo | [apache/dubbo](https://github.com/apache/dubbo) | `98e45385a6b41f120b5fbb2963b6f133c153b78a` | 0 | no retained warnings |
| test | rocketmq | [apache/rocketmq](https://github.com/apache/rocketmq) | `5ad6a3e5ceb4bca604866717c3bdcc61860f65e4` | 0 | no retained warnings |
| train | shardingsphere | [apache/shardingsphere](https://github.com/apache/shardingsphere) | `75474b1aa85f9f2e1ad72b2482fadb9dabb18253` | 0 | no retained warnings |
| train | questdb | [questdb/questdb](https://github.com/questdb/questdb) | `b6b3b15ba75ce118f58ee33b232af824bd0d5f1a` | 0 | no retained warnings |
| validation | trino | [trinodb/trino](https://github.com/trinodb/trino) | `ce0884ad4dbc2cc1f7f1e0fb644cd11612e92d0c` | 0 | no retained warnings |
| train | presto | [prestodb/presto](https://github.com/prestodb/presto) | `c2abc247f313c7d121e2da8f26d23d057681a5de` | 0 | no retained warnings |
| test | commons-jci | [apache/commons-jci](https://github.com/apache/commons-jci) | `97bbbb4263b6a79d1eb1614c4a30c416fbcd24d8` | 0 | no retained warnings |
| validation | opennlp | [apache/opennlp](https://github.com/apache/opennlp) | `6da32a6e72c0436074afc77de5db8da8b3b3f972` | 0 | no retained warnings |
| train | tomcat | [apache/tomcat](https://github.com/apache/tomcat) | `59a3e3932f8677bf24c56462b09e9dace586ef57` | 0 | no retained warnings |
| train | activemq | [apache/activemq](https://github.com/apache/activemq) | `58c82040fe5150fbbdee865563e360c73f0daa3d` | 0 | no retained warnings |
| train | ant | [apache/ant](https://github.com/apache/ant) | `125b2b9e2260ec1268dd73d75519fd146e8a5e6e` | 0 | no retained warnings |
| train | karaf | [apache/karaf](https://github.com/apache/karaf) | `433f8b7014fc764975e6016c0c6cfcb4dc89e554` | 0 | no retained warnings |
