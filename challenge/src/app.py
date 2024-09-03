import sys
import main.base.schema as schema
from main.job.pipeline import PySparkJob


eligibility_path = sys.argv[1]
medical_path = sys.argv[2]


def main():
    job = PySparkJob()

    # Load input data to DataFrame
    print("<<Reading>>")
    eligibility = job.read_csv(eligibility_path, schema.eligibility)
    medical = job.read_csv(medical_path, schema.medical)

    print("<<Filter>>")
    filtered_medicals = job.filter_medical(eligibility, medical)
    filtered_medicals.show()

    print("<<Full Name>>")
    full_names = job.generate_full_name(eligibility, filtered_medicals)
    full_names.show()

    print("<<Max Paid Member>>")
    max_paid_member = job.find_max_paid_member(filtered_medicals)
    print(max_paid_member)

    print("<<Total Paid Amount>>")
    total_paid_amount = job.find_total_paid_amount(filtered_medicals)
    print(total_paid_amount)
    job.stop()


if __name__ == '__main__':
    main()
