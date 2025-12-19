LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1MDECGiNk5Rskq-5TIduquavFiQrqZJoF' AS row
CALL(row) {
  WITH row,
    CASE WHEN row.src_label IS NULL OR TRIM(row.src_label) = "" THEN "Entity" ELSE TRIM(row.src_label) END AS src_label,
    CASE WHEN row.dst_label IS NULL OR TRIM(row.dst_label) = "" THEN "Entity" ELSE TRIM(row.dst_label) END AS dst_label

  CALL apoc.merge.node([src_label], {name: row.src}, {}) YIELD node AS a
  CALL apoc.merge.node([dst_label], {name: row.dst}, {}) YIELD node AS b
  CALL apoc.merge.relationship(a, row.rel, {}, {}, b) YIELD rel
  RETURN 1 as return_value
} IN TRANSACTIONS OF 1000 ROWS
RETURN count(*);
