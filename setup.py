from setuptools import setup 


setup(name="ml_util",
        version='0.1',
        description='utilties for machine learning with torch',
        url='no url yet',
        author='Kolten Pearson',
        author_email='koltenpearson@gmail.com',
        license='INTERNAL USE ONLY',
        packages=['ml_util'],
        scripts=['bin/mlarchive'],
        include_package_data=True,
        zip_safe=False)

