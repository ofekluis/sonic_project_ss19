
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def main():
	scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
	creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
	client = gspread.authorize(creds)
	sheet = client.open("SonicTable").sheet1  # Open the spreadhseet
	data = sheet.get_all_records()
	print (data)

if __name__ == "__main__":
    main()