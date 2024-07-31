import os

import dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

## Script to save dataset to db

dotenv.load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("DB_SERVER_URI")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class RakutenProducts(Base):
    __tablename__ = "rakuten_products"

    id = Column(Integer, primary_key=True, unique=True)
    designation = Column(String)
    description = Column(Text, nullable=True)
    productid = Column(String)
    imageid = Column(String)
    prdtypecode = Column(String, ForeignKey("prdtypecode_label_mapping.prdtypecode"))
    image_name = Column(String)

    prdtype = relationship("PrdtypecodeLabelMapping")


class PrdtypecodeLabelMapping(Base):
    __tablename__ = "prdtypecode_label_mapping"
    prdtypecode = Column(String, primary_key=True)
    label = Column(Integer, unique=True)


# Create tables
Base.metadata.create_all(engine)

# TODO: environment variable for path
dataset = pd.read_csv("data/preprocessed/dataset.csv", index_col=0)
dataset = dataset.fillna("")
dataset = dataset.drop("label", axis=1)
dataset[["productid", "imageid", "prdtypecode"]] = dataset[
    ["productid", "imageid", "prdtypecode"]
].astype("string")
dataset = dataset.reset_index(names=["id"])

# Encode product type code
label_encoder = LabelEncoder()
dataset["label"] = label_encoder.fit_transform(dataset["prdtypecode"])

# Save LabelEncoder mapping
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
mapping_dic = {
    str(class_): int(label)
    for class_, label in zip(
        label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
    )
}

session = SessionLocal()
try:
    for key, value in mapping_dic.items():
        mapping = PrdtypecodeLabelMapping(prdtypecode=key, label=value)
        session.add(mapping)

    session.commit()
except Exception as e:
    session.rollback()
    print(f"Error: {e}")
finally:
    session.close()

dataset = dataset.drop("label", axis=1)
data = dataset.to_dict(orient="records")
try:
    for row in data:
        existing_product = session.query(RakutenProducts).filter_by(id=row["id"]).first()

        if existing_product is None:
            product = RakutenProducts(**row)
            session.add(product)
        else:
            print(f"Duplicate found, skipping: {row['id']}")

    session.commit()
except Exception as e:

    session.rollback()
    print(f"Error: {e}")
finally:
    session.close()
#
#
