# Unseen Tool Generalization

Split: `all`.

Definition: A strict held-out-tool row has at least one gold tool id that never appears in the official train split gold plans.

- Train tasks: 4867
- Train gold tools: 497
- Train gold tool families: 259
- Eval tasks: 1998
- Eval non-empty workflows: 1303
- Eval unseen tool ids: 34
- Eval unseen tool families: 21
- Unseen source counts: {'tau2': 1, 'toolbench': 24}
- Unseen domain counts: {'mock': 1, 'validation': 24}

| method | overall n | overall tool EM | seen n | seen tool EM | unseen n | unseen tool EM | unseen schema valid | unseen tool recall | unseen family recall | unseen avg ED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM nearest | 1303 | 0.2671 | 1278 | 0.2723 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.2800 |
| FM constrained nearest | 1303 | 0.3200 | 1278 | 0.3263 | 25 | 0.0000 | 0.4400 | 0.0000 | 0.2000 | 2.3200 |
| FM AR available-mask | 1303 | 0.3208 | 1278 | 0.3271 | 25 | 0.0000 | 0.4000 | 0.0000 | 0.1600 | 2.2000 |
| FM optimized constrained | 1303 | 0.3101 | 1278 | 0.3161 | 25 | 0.0000 | 0.4400 | 0.0000 | 0.2000 | 2.7200 |

Interpretation: this is a workflow-planning generalization check, separate from tau2 closed-loop execution. Tool EM measures full workflow equality; unseen tool/family recall measures whether the decoder at least selects the held-out schema element; schema valid measures whether predicted tools stay inside the task's available-tool catalog.

Representative unseen tool ids:

- `tau2::mock::transfer_to_human_agents`
- `toolbench::validation::Search`
- `toolbench::validation::all_for_coupons`
- `toolbench::validation::anime_for_anime_manga_and_novels_api`
- `toolbench::validation::api_getcountries_for_similarweb_historical_data`
- `toolbench::validation::api_getvisits_for_similarweb_historical_data`
- `toolbench::validation::autocomplete_for_open_brewery_db`
- `toolbench::validation::balance_sheet_statement_for_financial_statements`
- `toolbench::validation::blacklist_phone_numbers_for_smsapi_com`
- `toolbench::validation::breweries_for_open_brewery_db`
- `toolbench::validation::cash_flow_statement_for_financial_statements`
- `toolbench::validation::commands_run_for_ssh_honeypot`
- `toolbench::validation::findalllistingactive_for_etsy`
- `toolbench::validation::get_airline_data_for_brazilian_airlines_real_flights_data`
- `toolbench::validation::get_novel_by_id_for_anime_manga_and_novels_api`
- `toolbench::validation::get_user_by_gender_for_fake_users`
- `toolbench::validation::getcompanies_for_get_360_business_tool`
- `toolbench::validation::getcompaniessince_for_get_360_business_tool`
- `toolbench::validation::getcompetitions_for_wosti_futbol_tv_peru`
- `toolbench::validation::getevents_for_wosti_futbol_tv_peru`
- `toolbench::validation::getgasprice_for_chaingateway_io`
- `toolbench::validation::getteams_for_wosti_futbol_tv_peru`
- `toolbench::validation::gst_for_gst_advance`
- `toolbench::validation::headphones_for_amazon_api_v2`
- `toolbench::validation::income_statement_for_financial_statements`
- `toolbench::validation::listaddresses_for_chaingateway_io`
- `toolbench::validation::login_data_for_ssh_honeypot`
- `toolbench::validation::mailboxvalidator_api_for_mailboxvalidator`
- `toolbench::validation::manga_for_anime_manga_and_novels_api`
- `toolbench::validation::novels_for_anime_manga_and_novels_api`

Representative unseen tool families:

- `tau2::mock::transfer_to_human_agents`
- `toolbench::validation::Search`
- `toolbench::validation::amazon_api_v2`
- `toolbench::validation::anime_manga_and_novels_api`
- `toolbench::validation::brazilian_airlines_real_flights_data`
- `toolbench::validation::chaingateway_io`
- `toolbench::validation::coupons`
- `toolbench::validation::etsy`
- `toolbench::validation::fake_users`
- `toolbench::validation::financial_statements`
- `toolbench::validation::get_360_business_tool`
- `toolbench::validation::gst_advance`
- `toolbench::validation::mailboxvalidator`
- `toolbench::validation::morning_star`
- `toolbench::validation::open_brewery_db`
- `toolbench::validation::similarweb_historical_data`
- `toolbench::validation::smsapi_com`
- `toolbench::validation::socialgrep`
- `toolbench::validation::ssh_honeypot`
- `toolbench::validation::wosti_futbol_tv_peru`
- `toolbench::validation::youtube_v3_v2`
