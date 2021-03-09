from shapash.utils.load_smartpredictor import load_smartpredictor


def main():
    predictor = load_smartpredictor('models/houseprices_predictor.pkl')
    app = predictor.run_app(host='localhost')


if __name__ == "__main__":
    main()
