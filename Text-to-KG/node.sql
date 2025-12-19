LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1r6qpkFSg5tkcX-bdgjpGDwkZIMZxLEsu' AS row
WITH TRIM(row.label) AS label, TRIM(row.name) AS name
WHERE label IS NOT NULL AND name IS NOT NULL AND label <> "" AND name <> ""
CALL apoc.create.node([label], {name: name}) YIELD node
RETURN count(*);
