# MLOps Lifecycle

## 1. Problem framing
- Define business goal and success metrics
- Identify data sources and constraints
Rent konkret:
- Det här kan ta mycket tid (flera veckor, många mail och möten)
- Measure twice cut once
- Definera ett affärsmål (t.ex förutspå en aktiekurs, klassificera kundfeedback, tolka sensordata)
- Definera en metric som kan beskriva framgång (t.ex. RMSE, F1-score, AUC, eller många andra beroende på problem)
     - För en aktiemodell kanske metricen är: "Hur mycket pengar tjänar den över en viss period?" (dvs. en ekonomisk metric snarare än en teknisk)
     - Vi kan kalla det en "business metric" eller "key performance indicator (KPI)" eller "meta-metric
- Identifiera datakällor (databaser, API:er, externa datasets)
     - Finns det historisk data? Är den av god kvalitet? Är den tillräckligt omfattande? Är den relevant för problemet?
     - Finns det några juridiska eller etiska begränsningar kring datan (t.ex. GDPR, bias)?
     - I historisk data kan det finnas mönster som vi inte vill fånga (hur tar vi hand om dem?)
    - (GPS + markdata [jordtyp, soltimmar, regn])

## 2. Data acquisition
- Data collection, labeling, and governance
- Version datasets and track provenance
Rent konkret:
- Hitta den faktiska datan
- Systematisera inhämtning, ETL (Extract - Transform - Load)
- Sätt korrekta labels (kan vara ett väldigt stort jobb)
- Att hålla kolla på metadata (t.ex vart/när datan kommer ifrån, när den ändrades osv) är en hel industri
    - Men vi får göra en glad ansats
- Systematisera versionshantering av data (t.ex DVC)
- För vårt problem:
    - Sätta upp en ETL-pipline från rå GPS-data, och externa markdatafiler till ett användbart format i en databas
    - Ideallt kanske vi använder ett data warehouse (specialliserad ändamålsenlig databas)
    - Vi skulle kunna använda Fivetran för att automatisera hämtandet av datan

## 3. Experimentation
- Feature engineering and model selection
- Reproducible experiments and tracking
Rent konkret:
- Vi bygger (i regel) notebooks, med massa EDA, experiment, visualiseringar
- Här är det bra att ordentligt sätta upp återskapningsbara miljöer/modeller osv
För vårt problem:
- Vi läser in datan från Data Warehouset in i en notebook
- Vi skapar relevanta .py-filer (för träning av modeller etc)
- Vi läser in funktioner från våra .py-filer
- Vi bygger odlingskvalitetsmodeller, utvärderar, plottar, experimenterar
- Vi sparar modeller, körningar, hyperparametrar osv, strukturerat och återskapningsbart

## 4. Training & validation
- Train models, evaluate against baselines
- Decide promotion criteria
Rent konkret: 
- Vi vill använda de återskapningsbara miljöer/modeller som vi tagit fram i experimentsteget
- Vi sparar träningsdata ihop med modeller, utvärderar kvalitet
- Ofta sparar vi succesivt den bästa modellen som baseline, och utvärderar sedan nya mot den
- När en modell blir tillräckligt mycket bättre än baseline, byter vi ut den
För vårt problem:
- Vi tränar odlingsmodellen, och förbereder den för produktionssättning! 

## 5. Packaging & deployment
- Package models (Docker, TorchScript, ONNX)
- CI/CD for model artifact
Rent konkret:
- Vi packeterar modellen (t.ex med docker), och sätter upp deploymentschemes
- Alltså: hur görs modellen tillgänglig i produktion? 
- Ett vanligt sätt är att när vi mergar till main (i git), så depolyar vi automatiskt

## 6. Monitoring
- Drift detection, performance metrics
- Alerts and rollback strategies

## 7. Continuous improvement
- Feedback loops from production
- Schedule retraining and audits

## Common roles
- Data engineer, ML engineer, MLOps engineer, product owner
