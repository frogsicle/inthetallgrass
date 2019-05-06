import argparse
import os
import sys
import random
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, UniqueConstraint, Boolean, create_engine


Base = declarative_base()


# table spec
class Image(Base):
    __tablename__ = "image"
    
    id = Column(Integer, primary_key=True)
    relative_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    is_test_set = Column(Boolean, nullable=False)
    label = Column(String)
    
    __table_args__ = (UniqueConstraint('file_name'),)

    
def mk_session(db_path='sqlite:///:memory:'):
    if not db_path.startswith('sqlite:///'):
        db_path = 'sqlite:///' + db_path
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def get_n_update_existing(session, file_name, label, relative_path):
    """updates _only_ label and relative_path if file_name already exists in db"""
    previous_entry = session.query(Image).filter(Image.file_name == file_name).first()
    if previous_entry is not None:
        logging.info('{} exists, updating (relative_path, label) from ({}, {}) to ({}, {})'.format(
            file_name, previous_entry.relative_path, previous_entry.label, relative_path, label))
        previous_entry.label = label
        previous_entry.relative_path = relative_path
    return previous_entry


def import_folder_with_label(base_dir, relative_path, label, session, update_duplicates=False, 
                             test_frac=0.3, ending=''):
    """enters image paths into db with label and test set assignment"""
    to_import = os.listdir('{}/{}/'.format(base_dir, relative_path))
    
    for file_name in to_import:
        if file_name.endswith(ending):
            image = None
            is_test = False
            if random.random() < test_frac:
                is_test = True
            if update_duplicates:
                # image will be None or an updated Image entry
                image = get_n_update_existing(session, file_name, label, relative_path)
            if image is None:
                image = Image(relative_path=relative_path, 
                              label=label,
                              file_name=file_name,
                              is_test_set=is_test)
            session.add(image)
    session.commit()
    
        
def main(args):
    assert 0 <= args.test_fraction <= 1.0, "given test_fraction {} not between 0. and 1.".format(args.test_fraction)
    session, engine = mk_session(args.db)

    import_folder_with_label(args.base_dir, args.relative_path, args.label, session, 
                             update_duplicates=args.update_duplicates, 
                             test_frac=args.test_fraction,
                             ending=args.select_by_ending)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="enter folder of images into db with label & is_test assignment")
    parser.add_argument('--base_dir', help='absolute path to data directory', required=True)
    parser.add_argument('--relative_path', help='relative path (from base_dir) to folder of images to import', required=True)
    parser.add_argument('--label', help='label (string) to assign to images', required=True)
    parser.add_argument('--db', help='path to sqlite3 db to update or create', required=True)
    parser.add_argument('--test_fraction', default=0.3, type=float, 
                        help='fraction of images to be assigned to the test set')
    parser.add_argument('--update_duplicates', action='store_true', 
                        help='updates/overwrites the relative directory and label of images with duplicated file names')
    parser.add_argument('--select_by_ending', default='', help='enter only files with given ending (e.g. ".jpg")')

    args = parser.parse_args()

    main(args)
