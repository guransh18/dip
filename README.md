# adbms

link - 
https://tanishhhhh.notion.site/Practical-Exam-1547836f7463806d971ccd89a4884836 

Practical Exam 
Practical 01: Oracle SQL
 Practical 02: ORDBMS
Practical 03: XML 
Practical 04: Temporal 
Practical 05: Spatial databases 
Practical 06: Vertical Fragmentation   
Practical 07: Horizontal Fragmentation 
Practical 08: MongoDB
Practical 09: Redis 
Working Codes
Practical 01:
--CUSTOMER,PRODUCT,DEPARTMENT,EMPLOYEE

--PRODUCT
CREATE TABLE PRODUCT(
    PRODUCT_NAME VARCHAR(20),
    PRODUCT_ID INT PRIMARY KEY,
    RELEASE_DATE DATE 
);

--PRODUCT DATA
INSERT ALL 
INTO PRODUCT(PRODUCT_NAME, PRODUCT_ID, RELEASE_DATE) VALUES('LAPTOP',1,TO_DATE('2020-12-24','YYYY-MM-DD'))
INTO PRODUCT(PRODUCT_NAME, PRODUCT_ID, RELEASE_DATE) VALUES('MOBILE',2,TO_DATE('2020-11-24','YYYY-MM-DD'))
INTO PRODUCT(PRODUCT_NAME, PRODUCT_ID, RELEASE_DATE) VALUES('TV',3,TO_DATE('2020-10-24','YYYY-MM-DD'))
INTO PRODUCT(PRODUCT_NAME, PRODUCT_ID, RELEASE_DATE) VALUES('FRIDGE',4,TO_DATE('2020-02-24','YYYY-MM-DD'))
INTO PRODUCT(PRODUCT_NAME, PRODUCT_ID, RELEASE_DATE) VALUES('AC',5,TO_DATE('2020-03-24','YYYY-MM-DD'))
SELECT * FROM DUAL;
SELECT * FROM PRODUCT;

--DEPARTMENT
CREATE TABLE DEPARTMENT (
    DEPARTMENT_NAME VARCHAR(20),
    DEPARTMENT_ID INT PRIMARY KEY,
    DEPARTMENT_BUDGET INT
);

--DEPARTMENT DATA 
INSERT ALL 
INTO DEPARTMENT(DEPARTMENT_NAME, DEPARTMENT_ID, DEPARTMENT_BUDGET) VALUES('SALES',1,10000000)
INTO DEPARTMENT(DEPARTMENT_NAME, DEPARTMENT_ID, DEPARTMENT_BUDGET) VALUES('ENGINEERING',2,20000000)
INTO DEPARTMENT(DEPARTMENT_NAME, DEPARTMENT_ID, DEPARTMENT_BUDGET) VALUES('MARKETING',3,30000000)
INTO DEPARTMENT(DEPARTMENT_NAME, DEPARTMENT_ID, DEPARTMENT_BUDGET) VALUES('R&D',4,40000000)
INTO DEPARTMENT(DEPARTMENT_NAME, DEPARTMENT_ID, DEPARTMENT_BUDGET) VALUES('FINANCE',5,50000000)
SELECT * FROM DUAL;
SELECT * FROM PRODUCT;

--EMPLOYEE 
CREATE TABLE EMPLOYEE (
    NAME VARCHAR(50),
    SKILL VARCHAR(20),
    YEARS_OF_EXPERIENCE INT,
    DEPARTMENT_ID  INT,
    PRODUCT_ID INT,
    FOREIGN KEY(DEPARTMENT_ID) REFERENCES DEPARTMENT(DEPARTMENT_ID),
    FOREIGN KEY(PRODUCT_ID) REFERENCES PRODUCT(PRODUCT_ID)
);

--EMPLOYEE DATA 
INSERT ALL 
INTO EMPLOYEE(NAME, SKILL, YEARS_OF_EXPERIENCE, DEPARTMENT_ID, PRODUCT_ID) VALUES('EMPLOYEE1','SALES',5,1,1)
INTO EMPLOYEE(NAME, SKILL, YEARS_OF_EXPERIENCE, DEPARTMENT_ID, PRODUCT_ID) VALUES('EMPLOYEE2','ENGINEERING',4,2,2)
INTO EMPLOYEE(NAME, SKILL, YEARS_OF_EXPERIENCE, DEPARTMENT_ID, PRODUCT_ID) VALUES('EMPLOYEE3','MARKETING',3,3,3)
INTO EMPLOYEE(NAME, SKILL, YEARS_OF_EXPERIENCE, DEPARTMENT_ID, PRODUCT_ID) VALUES('EMPLOYEE4','R&D',2,4,4)
INTO EMPLOYEE(NAME, SKILL, YEARS_OF_EXPERIENCE, DEPARTMENT_ID, PRODUCT_ID) VALUES('EMPLOYEE5','FINANCE',1,5,5)
SELECT * FROM DUAL;
SELECT * FROM EMPLOYEE;

--CUSTOMER 
CREATE TABLE CUSTOMER (
    CUSTOMER_NAME VARCHAR(20),
    CUSTOMER_ID NUMBER,
    CUSTOMER_ADDRESS VARCHAR(50),
    PRODUCT_ID NUMBER,
    FOREIGN KEY (PRODUCT_ID) REFERENCES PRODUCT(PRODUCT_ID)
);

INSERT ALL 
INTO CUSTOMER(CUSTOMER_NAME, CUSTOMER_ID, CUSTOMER_ADDRESS, PRODUCT_ID) VALUES('CUSTOMER1',1,'MUMBAI',1)
INTO CUSTOMER(CUSTOMER_NAME, CUSTOMER_ID, CUSTOMER_ADDRESS, PRODUCT_ID) VALUES('CUSTOMER2',2,'SURAT',2)
INTO CUSTOMER(CUSTOMER_NAME, CUSTOMER_ID, CUSTOMER_ADDRESS, PRODUCT_ID) VALUES('CUSTOMER3',3,'KERALA',3)
INTO CUSTOMER(CUSTOMER_NAME, CUSTOMER_ID, CUSTOMER_ADDRESS, PRODUCT_ID) VALUES('CUSTOMER4',4,'CHENNAI',4)
INTO CUSTOMER(CUSTOMER_NAME, CUSTOMER_ID, CUSTOMER_ADDRESS, PRODUCT_ID) VALUES('CUSTOMER5',5,'ODISA',5)
SELECT * FROM DUAL;
SELECT * FROM CUSTOMER;


​
Practical 02:
-- Address
CREATE OR REPLACE TYPE AddrType1 AS OBJECT (
    Pincode    NUMBER(5),
    Street     CHAR(20),
    City       VARCHAR2(20),
    State      VARCHAR2(40),
    No         NUMBER(4)
);

-- Branch
CREATE OR REPLACE TYPE BranchType AS OBJECT (
    Address  AddrType1,
    Phone1   INTEGER,
    Phone2   INTEGER
);

CREATE OR REPLACE TYPE BranchTableType AS TABLE OF BranchType;
NESTED TABLE Branches STORE AS BranchTable;

-- Author
CREATE OR REPLACE TYPE AuthorType AS OBJECT (
    Name    VARCHAR2(50),
    Addr    AddrType1
);
CREATE OR REPLACE TYPE AuthorListType AS VARRAY(10) OF REF AuthorType;
CREATE TABLE Authors OF AuthorType;
INSERT INTO Authors VALUES (
    'Tanish',
    AddrType1(343, 'Savarkar', 'Thane', 'Maharashtra', 34)
);

-- Publisher
CREATE OR REPLACE TYPE PublisherType AS OBJECT (
    Name     VARCHAR2(50),
    Address  AddrType1,
    Branches BranchTableType
);
CREATE TABLE Publishers OF PublisherType; -- Fix missing semicolon
INSERT INTO Publishers VALUES (
    'Tanish',
    AddrType1(4002, 'Park Street', 'Mumbai', 'Maharashtra', 03),
    BranchTableType(
        BranchType(
            AddrType1(5002, 'Fstreet', 'Mumbai', 'Maharashtra', 03),
            234234,
            3245434
        )
    )
);

-- Books
CREATE TABLE Books (
    Title         VARCHAR2(50),
    Year          DATE,
    Published_By  REF PublisherType,
    Authors       AuthorListType
);
INSERT INTO Books 
SELECT 
    'IP',
    TO_DATE('28-MAY-1983', 'DD-MON-YYYY'),
    REF(pub),
    AuthorListType(REF(aut))
FROM 
    Publishers pub, 
    Authors aut 
WHERE 
    pub.Name = 'Tanish' 
    AND aut.Name = 'Tanish';

-- Queries
-- a) List all of the authors that have the same pin code as their publisher:
SELECT a.Name AS Author_Name
FROM Authors a, Publishers p
WHERE a.Addr.Pincode = p.Address.Pincode;

-- b) List all books that have 2 or more authors:
SELECT b.Title AS Book_Title
FROM Books b
WHERE CARDINALITY(b.Authors) >= 2;

-- c) List the name of the publisher that has the most branches:
SELECT p.Name AS Publisher_Name
FROM Publishers p
WHERE CARDINALITY(p.Branches) = (
    SELECT MAX(CARDINALITY(p1.Branches))
    FROM Publishers p1
);

-- d) Name of authors who have not published a book:
SELECT a.Name AS Author_Name
FROM Authors a
WHERE NOT EXISTS (
    SELECT 1 
    FROM Books b 
    WHERE a.OBJECT_ID = ANY(SELECT REF(aut) FROM TABLE(b.Authors) aut)
);

-- e) Name of authors who have not published a book (alternative approach):
SELECT a.Name AS Author_Name
FROM Authors a
WHERE a.Name NOT IN (
    SELECT au.Name
    FROM Books b, TABLE(b.Authors) au
);

​
Practical 03 XML:
--CREATE 
CREATE TABLE EMPLOYEE(
    EMPLOYEE_ID NUMBER(4),
    EMPLOYEE XMLTYPE
);

INSERT INTO EMPLOYEE VALUES (1, XMLTYPE(
    '<EMPLOYEE ID = "1">
    <NAME>TANISH1</NAME>
    <EMAIL>TANISH@TANISH.COM</EMAIL>
    <ACC_NO>12345</ACC_NO>
    <DOJ>12/09/20005</DOJ>
    </EMPLOYEE>'))

--READ
SELECT 
    EXTRACT(E.EMPLOYEE, '/EMPLOYEE/NAME/text()').getStringVal() AS NAME,
    EXTRACT(E.EMPLOYEE, '/EMPLOYEE/ACC_NO/text()').getStringVal() AS ACC_NO,
	  EXTRACT(E.EMPLOYEE, '/EMPLOYEE/EMAIL/text()').getStringVal() AS EMAIL,
    EXTRACT(E.EMPLOYEE, '/EMPLOYEE/DOJ/text()').getStringVal() AS DOJ
FROM 
    EMPLOYEE E;
    
--UPDATE 
UPDATE EMPLOYEE w
SET EMPLOYEE = XMLTYPE('<EMPLOYEE ID = "1">
    <NAME>TANISH1</NAME>
    <EMAIL>TANISH@TANISH.COM</EMAIL>
    <ACC_NO>1234567</ACC_NO>
    <DOJ>12/09/20005</DOJ>
    </EMPLOYEE>')
WHERE w.EMPLOYEE.EXTRACT('/EMPLOYEE/ACC_NO/text()').getStringVal() = '12345' 

--DELETE
DELETE FROM EMPLOYEE w
WHERE w.EMPLOYEE.EXTRACT('/EMPLOYEE/ACC_NO/text()').getStringVal() = '12345' 
​
Practical 04 TEMPORAL:
-- CREATE TABLE
CREATE TABLE SHARES (
    CUSTOMER_NAME VARCHAR(20),
    NUMBER_OF_SHARES NUMBER(10),
    PRICE_PER_SHARE NUMBER(5),
    TRANSACTION_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- INSERT DATA
INSERT ALL 
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE) VALUES ('COMPANY_A', 100, 100)
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE) VALUES ('COMPANY_B', 200, 200)
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE) VALUES ('COMPANY_C', 300, 300)
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE) VALUES ('COMPANY_D', 400, 400)
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE) VALUES ('COMPANY_E', 500, 500)
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE) VALUES ('COMPANY_F', 600, 600)
SELECT * FROM DUAL;

-- VERIFY INSERTED DATA
SELECT * FROM SHARES;

-- INSERT DATA WITH TIMESTAMP
INSERT ALL
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE, TRANSACTION_TIME) 
    VALUES ('COMPANY_A', 100, 100, TIMESTAMP '2024-12-12 11:45:00.000000')
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE, TRANSACTION_TIME) 
    VALUES ('COMPANY_B', 100, 100, TIMESTAMP '2024-11-09 01:15:00.000000')
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE, TRANSACTION_TIME) 
    VALUES ('COMPANY_C', 100, 100, TIMESTAMP '2024-08-07 04:00:00.000000')
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE, TRANSACTION_TIME) 
    VALUES ('COMPANY_B', 100, 100, TIMESTAMP '2024-01-06 02:15:00.000000')
    INTO SHARES (CUSTOMER_NAME, NUMBER_OF_SHARES, PRICE_PER_SHARE, TRANSACTION_TIME) 
    VALUES ('COMPANY_B', 100, 100, TIMESTAMP '2024-04-08 01:35:00.000000')
SELECT * FROM DUAL;

-- VERIFY INSERTED DATA WITH TIMESTAMP
SELECT * FROM SHARES;

-- QUERY 01: Select records where the price per share is greater than 15 and the transaction time is exactly '11:45:00'
SELECT CUSTOMER_NAME 
FROM SHARES 
WHERE PRICE_PER_SHARE > 15 
AND TO_CHAR(TRANSACTION_TIME, 'HH24:MI:SS') = '11:45:00';

-- QUERY 02: Select records where the price per share is the highest and the transaction time is exactly '01:35:00'
SELECT CUSTOMER_NAME 
FROM SHARES 
WHERE PRICE_PER_SHARE IN (
    SELECT MAX(PRICE_PER_SHARE) 
    FROM SHARES 
    WHERE TO_CHAR(TRANSACTION_TIME, 'HH24:MI:SS') = '01:35:00'
);

CREATE TABLE STOCK_PRICES (
    COMPANY_NAME VARCHAR(50),
    PRICE_PER_SHARE NUMBER(10,2),
    VALID_FROM TIMESTAMP,
    VALID_TO TIMESTAMP,
    PRIMARY KEY(COMPANY_NAME, VALID_FROM)
);

INSERT ALL 
    INTO STOCK_PRICES(COMPANY_NAME, PRICE_PER_SHARE, VALID_FROM, VALID_TO) 
    VALUES('A', 100.5, TIMESTAMP '2023-02-03 00:00:00.000000', TIMESTAMP '2023-06-03 23:24:25.000000')
    INTO STOCK_PRICES(COMPANY_NAME, PRICE_PER_SHARE, VALID_FROM, VALID_TO) 
    VALUES('B', 110.5, TIMESTAMP '2023-03-03 01:00:00.000000', TIMESTAMP '2023-07-03 22:20:20.000000')
    INTO STOCK_PRICES(COMPANY_NAME, PRICE_PER_SHARE, VALID_FROM, VALID_TO) 
    VALUES('C', 120.5, TIMESTAMP '2023-04-03 00:00:00.000000', TIMESTAMP '2023-08-03 21:19:18.000000')
    INTO STOCK_PRICES(COMPANY_NAME, PRICE_PER_SHARE, VALID_FROM, VALID_TO) 
    VALUES('D', 130.5, TIMESTAMP '2023-05-03 00:00:00.000000', TIMESTAMP '2023-09-03 20:17:13.000000')
    INTO STOCK_PRICES(COMPANY_NAME, PRICE_PER_SHARE, VALID_FROM, VALID_TO) 
    VALUES('E', 140.5, TIMESTAMP '2023-06-03 00:00:00.000000', TIMESTAMP '2023-10-03 15:14:12.000000')
SELECT * FROM DUAL;

SELECT * FROM STOCK_PRICES;

--Querying the temporal database: 
SELECT COMPANY_NAME, PRICE_PER_SHARE
FROM STOCK_PRICES
WHERE COMPANY_NAME = 'A'
AND SYSDATE BETWEEN VALID_FROM AND VALID_TO;

--Update stock prices:
set valid_to = timestamp '2023-12-31, 23:59:59'
whrere comapny_name = 'CompanyA'
and valid_to = timestamp '9999-12-31 23:59:59'; 

--Insert the new stock price for Company A
insert into stock_prices(company_name, price_per share, valid_from, valid_to)
values ('CompanyA', 110.25, timestamp '2024-01-01 00:00:00', timestamp '9999-12-31 23:59:59);

--Get the full history of stock prices for Company A
select company_name, price_per_share, valid_from, valid_to
from stock_prices
where company_name = 'ComapnyA'
order by valid_from;
​
Practical 05 SPATIAL:
-- Create the table to store market data with spatial geometry
CREATE TABLE cola_markets1(
    Mkt_id NUMBER PRIMARY KEY,
    Name VARCHAR(32),
    Shape MDSYS.SDO_GEOMETRY
);

-- First Row: Inserting data for market 'abc' with geometry
INSERT INTO cola_markets1 VALUES (1, 'abc', MDSYS.SDO_GEOMETRY(2003, NULL, NULL, MDSYS.SDO_ELEM_INFO_ARRAY(1, 1003, 3), MDSYS.SDO_ORDINATE_ARRAY(1, 1, 5, 7)));

-- Second Row: Inserting data for market 'pqr' with geometry
INSERT INTO cola_markets1 VALUES (2, 'pqr', MDSYS.SDO_GEOMETRY(2003, NULL, NULL, MDSYS.SDO_ELEM_INFO_ARRAY(1, 1003, 1), MDSYS.SDO_ORDINATE_ARRAY(5, 1, 8, 1, 8, 6, 5, 7, 5, 1)));

-- Third Row: Inserting data for market 'mno' with geometry
INSERT INTO cola_markets1 VALUES (3, 'mno', MDSYS.SDO_GEOMETRY(2003, NULL, NULL, MDSYS.SDO_ELEM_INFO_ARRAY(1, 1003, 1), MDSYS.SDO_ORDINATE_ARRAY(3, 3, 6, 3, 6, 5, 4, 5, 3, 3)));

-- Fourth Row: Inserting data for market 'xyz' with geometry
INSERT INTO cola_markets1 VALUES (4, 'xyz', MDSYS.SDO_GEOMETRY(2003, NULL, NULL, MDSYS.SDO_ELEM_INFO_ARRAY(1, 1003, 4), MDSYS.SDO_ORDINATE_ARRAY(8, 7, 10, 9, 8, 11)));

-- Fifth Row: Inserting metadata for the spatial column 'shape'
INSERT INTO user_sdo_geom_metadata (table_name, column_name, diminfo, srid) 
VALUES ('cola_markets1', 'shape', MDSYS.SDO_DIM_ARRAY(MDSYS.SDO_DIM_ELEMENT('X', 0, 20, 0.005), MDSYS.SDO_DIM_ELEMENT('Y', 0, 20, 0.005)), NULL);

-- Creating spatial index on 'shape' column for optimization
CREATE INDEX cola_spatial_idx
ON cola_markets1(shape)
INDEXTYPE IS MDSYS.SPATIAL_INDEX;

-- Query to find the intersection of two market shapes
SELECT sdo_geom.sdo_intersection(c_a.shape, c_c.shape, 0.005)
FROM cola_markets1 c_a, cola_markets1 c_c
WHERE c_a.name = 'abc' AND c_c.name = 'mno';

-- Query to check if two market shapes are equal within a tolerance
SELECT sdo_geom.relate(c_b.shape, 'equal', c_d.shape, 0.005)
FROM cola_markets1 c_b, cola_markets1 c_d
WHERE c_b.name = 'abc' AND c_d.name = 'mno';

-- Query to calculate the area of each market shape
SELECT name, sdo_geom.sdo_area(shape, 0.005)
FROM cola_markets1;

-- Query to calculate the area of a specific market shape 'xyz'
SELECT c.name, sdo_geom.sdo_area(c.shape, 0.005)
FROM cola_markets1 c WHERE c.name = 'xyz';

-- Query to calculate the distance between two market shapes
SELECT sdo_geom.sdo_distance(c_b.shape, c_d.shape, 0.005)
FROM cola_markets1 c_b, cola_markets1 c_d
WHERE c_b.name = 'abc' AND c_d.name = 'xyz';

​
Practical 06: VERTICAL FRAGMNENTATAION
--VERTICAL FRAGMNENTATAION

--GLOBAL TABLE STUDENT 
CREATE TABLE STUDENT (
    ROLL_NO VARCHAR(20) PRIMARY KEY,
    NAME VARCHAR(50),
    DEPARTMENT VARCHAR(10),
    ADDITIONAL_COURSE VARCHAR(20),
    FEES_PAID NUMBER,
    ADDRESS VARCHAR (50)
);

-- Create user C##USER1:
CREATE USER C##USER1 IDENTIFIED BY password;
GRANT CONNECT, RESOURCE TO C##USER1;

-- Create user C##USER2:
CREATE USER C##USER2 IDENTIFIED BY password;
GRANT CONNECT, RESOURCE TO C##USER2;
GRANT UNLIMITED TABLESPACE TO C##USER1, C##USER2;

-- Connect to C##USER1
CONNECT C##USER1/password;
CREATE TABLE student1 (
 roll_no INT PRIMARY KEY,
 name VARCHAR(50),
 address VARCHAR(100)
);

-- Connect to C##USER2
CONNECT C##USER2/password;
CREATE TABLE student2 (
 roll_no INT PRIMARY KEY,
 department VARCHAR(50),
 additional_course VARCHAR(50),
 fees_paid DECIMAL(10, 2)
);

--DATA INSERT 
INSERT ALL 
    INTO STUDENT(ROLL_NO, NAME, DEPARTMENT, ADDITIONAL_COURSE,FEES_PAID, ADDRESS) 
    VALUES(001,'STUDENT1','CS','FINANCE',25000,'MUMBAI')
    INTO STUDENT(ROLL_NO, NAME, DEPARTMENT, ADDITIONAL_COURSE,FEES_PAID, ADDRESS) 
    VALUES(002,'STUDENT2','IT','AI',35000,'DELHI')
    INTO STUDENT(ROLL_NO, NAME, DEPARTMENT, ADDITIONAL_COURSE,FEES_PAID, ADDRESS) 
    VALUES(003,'STUDENT3','BBA','POWERBI',45000,'SURAT')
    INTO STUDENT(ROLL_NO, NAME, DEPARTMENT, ADDITIONAL_COURSE,FEES_PAID, ADDRESS) 
    VALUES(004,'STUDENT4','BBM','MARKET',55000,'THANE')
    INTO STUDENT(ROLL_NO, NAME, DEPARTMENT, ADDITIONAL_COURSE,FEES_PAID, ADDRESS) 
    VALUES(005,'STUDENT5','BFM','SOMETHING',65000,'PANVEL')
SELECT * FROM DUAL;

--STUDENT1
CONNECT C##USER1/password;
INSERT ALL 
    INTO STUDENT1(ROLL_NO,NAME,ADDRESS) 
    VALUES(001,'STUDENT1','MUMBAI')
    INTO STUDENT1(ROLL_NO, NAME,ADDRESS) 
    VALUES(002,'STUDENT2','DELHI')
    INTO STUDENT1(ROLL_NO, NAME,ADDRESS) 
    VALUES(003,'STUDENT3','SURAT')
    INTO STUDENT1(ROLL_NO, NAME,ADDRESS) 
    VALUES(004,'STUDENT4','THANE')
    INTO STUDENT1(ROLL_NO, NAME,ADDRESS) 
    VALUES(005,'STUDENT5','PANVEL')
SELECT * FROM DUAL;

--STUDENT2
CONNECT C##USER2/password;
INSERT ALL 
   	INTO STUDENT2(NAME, DEPARTMENT, ADDITIONAL_COURSE, FEES_PAID) 
	VALUES('STUDENT1','CS','FINANCE',25000)
    INTO STUDENT2(NAME, DEPARTMENT, ADDITIONAL_COURSE, FEES_PAID) 
	VALUES('STUDENT2','IT','AI',35000)
    INTO STUDENT2(NAME, DEPARTMENT, ADDITIONAL_COURSE, FEES_PAID) 
	VALUES('STUDENT3','BBA','POWERBI',45000)
    INTO STUDENT2(NAME, DEPARTMENT, ADDITIONAL_COURSE, FEES_PAID) 
	VALUES('STUDENT4','BBM','MARKET',55000)
    INTO STUDENT2(NAME, DEPARTMENT, ADDITIONAL_COURSE, FEES_PAID) 
	VALUES('STUDENT5','BFM','SOMETHING',65000)
SELECT * FROM DUAL;

-- A] Display all records in the student table
CONNECT SYSTEM;
-- Join tables from C##USER1 and C##USER2
SELECT s1.roll_no, s1.name, s1.address, s2.department, s2.additional_course, s2.fees_paid
FROM C##USER1.student1 s1
JOIN C##USER2.student2 s2 ON s1.roll_no = s2.roll_no;

-- B] Display additional course name for student roll no 002
SELECT additional_course 
FROM C##USER2.student2 
WHERE roll_no = 3;

-- C] Display fees paid for a given student name and roll no
SELECT s2.fees_paid
FROM C##USER1.student1 s1
JOIN C##USER2.student2 s2 ON s1.roll_no = s2.roll_no
WHERE s1.name = 'Jane Smith';

-- D] Display name of all students whose fees paid is greater than a given value (e.g., 1000)
SELECT s1.name
FROM C##USER1.student1 s1
JOIN C##USER2.student2 s2 ON s1.roll_no = s2.roll_no
WHERE s2.fees_paid > 50000;
​
Practical 07: HORIZZONTAL FRAGMNENTATAION
--TABLE BOOK
CREATE TABLE Book(id INT, name VARCHAR2(10));

--INSERT DATA 
INSERT INTO Book VALUES(1, 'Python');
INSERT INTO Book VALUES(2, 'Database');
INSERT INTO Book VALUES(3, 'Big Data');
INSERT INTO Book VALUES(4, 'Java');
INSERT INTO Book VALUES(5, 'C++');

--DISPLAY 
SELECT * FROM Book;

--CREATE A USER AND GRANT PRIVILEGES
CREATE USER MyMSC IDENTIFIED BY MyMSC;
GRANT CREATE SESSION TO MyMSC;
GRANT CREATE TABLE TO MyMSC;
GRANT UNLIMITED TABLESPACE TO MyMSC;
GRANT CREATE DATABASE LINK TO MyMSC;

--CONNECT WITH THE MYMSC USER
CONNECT C##MyMSC;

--CREATE A DATABASE LINK
CREATE DATABASE LINK Link1 CONNECT TO system IDENTIFIED BY system;

--CONNECT WITH THE SYSTEM USER:
CONNECT system;

--CHECK HOSTNAME
SELECT HOST_NAME FROM v$instance;

--PUBLIC LINK
CREATE PUBLIC DATABASE LINK Link001 CONNECT TO system IDENTIFIED BY system USING 'XE';
SELECT * FROM Book@Link001;

--NEW TABLE 
CREATE TABLE Book1(id INT, name VARCHAR2(10));
INSERT INTO Book1 VALUES(3, 'Big Data');
INSERT INTO Book1 VALUES(4, 'Data Sci');
INSERT INTO Book1 VALUES(5, 'AI');
INSERT INTO Book1 VALUES(6, 'ML');
INSERT INTO Book1 VALUES(7, 'Cloud');
SELECT * FROM Book1;

SELECT * FROM Book1 UNION SELECT * FROM Book@Link001;
​
Practical 08: MONOGDB
// Switch to the 'electronics_store' database
use electronics_store;

// Create a collection called 'products'
db.createCollection("products");

// Inserting multiple products
db.products.insertMany([
 {
 prod_name: "Laptop Pro X",
 category: "Laptop",
 price: 120000,
 stock: 50,
 manufacturer: { name: "TechCorp", country: "USA" }
 },
 {
 prod_name: "Smartphone Z5",
 category: "Smartphone",
 price: 50000,
 stock: 200,
 manufacturer: { name: "MobileTech", country: "China" }
 },
 {
 prod_name: "Smartwatch S8",
 category: "Wearables",
 price: 15000,
 stock: 120,
 manufacturer: { name: "WearableWorks", country: "Germany" }
 }
]);

// Query all products in the 'Smartphone' category
db.products.find({ category: "Smartphone" })

// Update the price of 'Smartwatch S8' to 18000
db.products.updateOne(
 { prod_name: "Smartwatch S8" },
 { $set: { price: 18000 } }
);

// Delete the product 'Laptop Pro X'
db.products.deleteOne({ prod_name: "Laptop Pro X" });

// Query products manufactured by 'MobileTech'
db.products.find({ "manufacturer.name": "MobileTech" });

// Query products with a price less than 20000
db.products.find({ price: { $lt: 20000 } });

// Sort products by stock in descending order
db.products.find().sort({ stock: -1 });


​
Practical 09: Redis
--ToDo List
LPUSH todo:list "Buy groceries"
LPUSH todo:list "Read a book"
LPUSH todo:list "Walk the dog"
LRANGE todo:list 0 -1
LPOP todo:list
LRANGE todo:list 0 -1
LRANGE completed:list 0 -1
LREM todo:list 0 "Read a book"
 
 --example 02
SET name 'John'

GET name

INCR counter

RPUSH users "Alice"

LRANGE users 0 -1
LLEN users
LRANGE users 0 -1
RPUSH users "Bob"
RPUSH users "Sara"
LLEN users
LRANGE users 0 -1

HSET product:101 name "Laptop" price 800 stock 10 category "Electronics"
HSET product:102 name "Phone" price 500 stock 25 category "Electronics"
HSET product:103 name "Headphones" price 100 stock 50 category "Accessories"

HGETALL product:101

ZADD product_prices 800 "product:101" 500 "product:102" 100 "product:103"
ZRANGEBYSCORE product_prices -inf +inf WITHSCORES

ZADD prices 800 "product:101" 500 "product:102" 100 "product:103"
ZRANGEBYSCORE prices 100 600 WITHSCORES

HSET product:101 price 750
HSET product:102 stock 20

DEL product:101
