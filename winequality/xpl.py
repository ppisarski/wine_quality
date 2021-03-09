from shapash.explainer.smart_explainer import SmartExplainer


def main():
    xpl = SmartExplainer()
    xpl.load('models/winequality-red_xpl.pkl')
    app = xpl.run_app(host='localhost')


if __name__ == "__main__":
    main()
