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


class Portfolio(db.Model):
    __tablename__ = "portfolios"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(12))

    def __repr__(self):
        return "<Portfolio(name={self.name!r})>".format(self=self)


class Company(db.Model):
    __tablename__ = "companies"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(12))
    exchange = db.Column(db.String(12))
    name = db.Column(db.String(12))
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolios.id"))
    portfolio = db.relationship("Portfolio", backref="companies")
    daloopa_company_id = db.Column(db.Integer, db.ForeignKey("daloopa_companies.id"))
    daloopa_company = db.relationship("DaloopaCompany", backref="companies")

    def __repr__(self):
        return "<Company(name={self.ticker!r})>".format(self=self)


class DaloopaCompany(db.Model):
    __tablename__ = "daloopa_companies"
    id = db.Column(db.Integer, primary_key=True)
    daloopa_ticker = db.Column(db.String(12))


# class DaloopaData(db.Model):
#     __tablename__ = "daloopa_data"
#     id = db.Column(db.Integer, primary_key=True)
#     ticker = db.Column(db.String(12))
#     company_name = db.Column(db.String(12))
#     label = db.Column(db.String(250))
#     category = db.Column(db.String(250))
#     span = db.Column(db.String(12))
#     calendar_period = db.Column(db.String(12))
#     fiscal_period = db.Column(db.String(12))
#     fiscal_date = db.Column(db.String(12))
#     unit = db.Column(db.String(12))
#     filing_type = db.Column(db.String(12))
#     value_raw = db.Column(db.String(50))
#     value_normalized = db.Column(db.String(50))
#     source_link = db.Column(db.String(250))
#     series_id = db.Column(db.String(12))
#     filing_date = db.Column(db.String(12))
#     series_id_relations = db.Column(db.String(12))
#     series_tag = db.Column(db.String(12))
#     restated = db.Column(db.String(12))
#     title = db.Column(db.String(250))
#     capiq_ticker = db.Column(db.String(12))
#     is_transition_period = db.Column(db.String(12))


class CompanySchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Company
        load_instance = True
        include_relationships = True
        include_fk = True


class PortfolioSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Portfolio
        load_instance = True
        include_relationships = True

    companies = ma.Nested(CompanySchema, only=("ticker", "exchange", "name"), many=True)


class DaloopaCompanySchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = DaloopaCompany
        load_instance = True
        include_relationships = True

    companies = ma.Nested(CompanySchema, only=("ticker", "exchange", "name"), many=True)


@app.route("/")
def index():
    one_company = Portfolio.query.all()
    portfolio_schema = PortfolioSchema(many=True)
    res2 = portfolio_schema.dump(one_company)

    one_company = DaloopaCompany.query.all()
    daloopa_schema = DaloopaCompanySchema(many=True)
    res3 = daloopa_schema.dump(one_company)

    return jsonify({"portfolios": res2, "daloopa_mappings": res3})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5150)
    # app.run(debug=True)
