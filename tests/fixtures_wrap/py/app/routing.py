from app.tickets import triage_ticket


def route(ticket, schema):
    return triage_ticket(ticket, schema)
