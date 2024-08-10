import os

import dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

## Script to save dataset to db

dotenv.load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("DB_SERVER_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class RakutenProducts(Base):
    __tablename__ = "rakuten_products"

    id = Column(Integer, primary_key=True, unique=True)
    designation = Column(String)
    description = Column(Text, nullable=True)
    text = Column(Text)
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


# Dataset
X_train = pd.read_csv("X_train_update.csv", index_col=0)
y_train = pd.read_csv("Y_train_CVw08PX.csv", index_col=0)
dataset = pd.concat([X_train, y_train], axis=1)

dataset = dataset.fillna("")
# Create text column
dataset["text"] = dataset["designation"] + " " + dataset["description"]

dataset[["productid", "imageid", "prdtypecode"]] = dataset[
    ["productid", "imageid", "prdtypecode"]
].astype("string")

# Create image name column
# of format image_1263597046_product_3804725264.jpg
dataset["image_name"] = "image_" + dataset["imageid"] + "_product_" + dataset["productid"]

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
# create mapping table
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

# create products table
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
