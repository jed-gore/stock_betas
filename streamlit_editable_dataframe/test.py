# serialize python objects to json

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
app.app_context().push()
ma = Marshmallow(app)


class Company(db.Model):
    __tablename__ = "companies"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(12))

    def __repr__(self):
        return "<Company(name={self.ticker!r})>".format(self=self)


class KPI(db.Model):
    __tablename__ = "kpis"
    id = db.Column(db.Integer, primary_key=True)
    kpi_name = db.Column(db.String(250))
    company_id = db.Column(db.Integer, db.ForeignKey("companies.id"))
    company = db.relationship("Company", backref="kpis")


class CompanySchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Company
        load_instance = True
        include_relationships = True


class KPISchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = KPI
        include_fk = True
        load_instance = True


@app.route("/")
def index():
    one_company = Company.query.all()
    company_schema = CompanySchema(many=True)
    res = company_schema.dump(one_company)
    return jsonify({"company": res})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5150)
    # app.run(debug=True)
