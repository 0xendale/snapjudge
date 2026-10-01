from app.triage import triage


def triage_ticket(ticket, schema):
    return triage(ticket.body, schema)
