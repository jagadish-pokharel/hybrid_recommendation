# your_app/management/commands/replace_genres.py
from django.core.management.base import BaseCommand
from tweets.models import Genre

class Command(BaseCommand):
    help = 'Replaces existing genres with a new set.'

    def handle(self, *args, **options):
        new_genres = [
            'Social Sciences', 'Fiction', 'History', 'Nature', 'Humor',
            'Cooking', 'Reference', "Children's Fiction", 'Health and Fitness',
            'Science', 'Politics', 'Biographies and Memoirs',
            'Business and Economics', 'Self-Help', 'Religion',
            'Mystery and Detective', 'Poetry', 'Language Arts',
            "Children's Nonfiction", 'Computers and Technology', 'Psychology',
            'Mind, Body, and Spirit', 'Family and Relationships', 'Philosophy',
            'Performing Arts', 'Travel', 'Drama', 'Literary Criticism',
            'True Crime'
        ]

        # Delete all existing genres
        Genre.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('All existing genres deleted.'))

        # Add the new genres
        for genre_name in new_genres:
            Genre.objects.create(name=genre_name)
            self.stdout.write(self.style.SUCCESS(f'Added genre: {genre_name}'))

        self.stdout.write(self.style.SUCCESS('Genres replaced successfully.'))