import csv

with open('dataset/datasetrag/cleaned_observations_with_tally.csv', 'r', encoding='utf-8') as csv_file, open('dataset/datasetrag/cleaned_observations_with_tally.txt', 'w', encoding='utf-8') as txt_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        observation_id = row['observation_id']
        species_name = row['species_name']
        common_name = row['common_name']
        date_observed = row['date_observed']
        location = row['location']
        image_url = row['image_url']
        notes = row['notes'] if row['notes'] else 'no notes provided'
        observed_count = row['observed_count']
        sentence = (
            f"On {date_observed}, {observed_count} individual(s) of the species "
            f"*{species_name}* (commonly known as {common_name}) were observed at coordinates {location}. "
            f"The observation (ID: {observation_id}) is documented with an image available at {image_url}. "
            f"Notes: {notes}.\n"
        )
        txt_file.write(sentence)