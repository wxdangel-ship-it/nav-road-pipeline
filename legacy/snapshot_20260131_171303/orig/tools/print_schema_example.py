import json

from pipeline.evidence.schema import example_candidate_record, example_primitive_record


def main() -> None:
    print("# PrimitiveEvidence")
    print(json.dumps(example_primitive_record(), ensure_ascii=True, indent=2))
    print("")
    print("# WorldCandidate")
    print(json.dumps(example_candidate_record(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
