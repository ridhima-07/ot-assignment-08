-- TPC-H schema (PostgreSQL). Baseline physical design = primary keys only.
DROP TABLE IF EXISTS lineitem, orders, partsupp, part, supplier, customer, nation, region CASCADE;
CREATE TABLE region   (r_regionkey int PRIMARY KEY, r_name char(25), r_comment varchar(152));
CREATE TABLE nation   (n_nationkey int PRIMARY KEY, n_name char(25), n_regionkey int, n_comment varchar(152));
CREATE TABLE part     (p_partkey int PRIMARY KEY, p_name varchar(55), p_mfgr char(25), p_brand char(10), p_type varchar(25),
                       p_size int, p_container char(10), p_retailprice numeric(15,2), p_comment varchar(23));
CREATE TABLE supplier (s_suppkey int PRIMARY KEY, s_name char(25), s_address varchar(40), s_nationkey int, s_phone char(15),
                       s_acctbal numeric(15,2), s_comment varchar(101));
CREATE TABLE partsupp (ps_partkey int, ps_suppkey int, ps_availqty int, ps_supplycost numeric(15,2), ps_comment varchar(199),
                       PRIMARY KEY (ps_partkey, ps_suppkey));
CREATE TABLE customer (c_custkey int PRIMARY KEY, c_name varchar(25), c_address varchar(40), c_nationkey int, c_phone char(15),
                       c_acctbal numeric(15,2), c_mktsegment char(10), c_comment varchar(117));
CREATE TABLE orders   (o_orderkey int PRIMARY KEY, o_custkey int, o_orderstatus char(1), o_totalprice numeric(15,2), o_orderdate date,
                       o_orderpriority char(15), o_clerk char(15), o_shippriority int, o_comment varchar(79));
CREATE TABLE lineitem (l_orderkey int, l_partkey int, l_suppkey int, l_linenumber int, l_quantity numeric(15,2),
                       l_extendedprice numeric(15,2), l_discount numeric(15,2), l_tax numeric(15,2), l_returnflag char(1),
                       l_linestatus char(1), l_shipdate date, l_commitdate date, l_receiptdate date, l_shipinstruct char(25),
                       l_shipmode char(10), l_comment varchar(44), PRIMARY KEY (l_orderkey, l_linenumber));
