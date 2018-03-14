from setuptools import setup 


setup(name="lantern",
        version='0.2',
        description='utilties for machine learning with torch',
        url='no url yet',
        author='Kolten Pearson',
        author_email='koltenpearson@gmail.com',
        license='INTERNAL USE ONLY',
        packages=['lantern', 'lantern.data', 'lantern.vis'],
        scripts=['bin/lantern'],
        include_package_data=True,
        zip_safe=False)

